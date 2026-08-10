import os
import sys

import torch
import streamlit as st
from transformers import TextIteratorStreamer

sys.path.insert(0, os.getcwd())
from scripts.Deploy.model_loader import ModelConfig, init_model
from scripts.Deploy.web_engine import LocalParams, api_generate, local_generate
from scripts.Deploy.web_utils import MODEL_PATHS, NATIVE_WEIGHTS, process_assistant_content, resolve_model_path, seed_generation


def _require_streamlit_run():
    """web_demo 是 Streamlit 应用，必须用 `streamlit run` 启动（裸 `python` 跑没有运行时）。"""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit = get_script_run_ctx() is not None
    except Exception:
        is_streamlit = False
    if not is_streamlit:
        print("错误：web_demo.py 是 Streamlit 应用，请用以下命令启动：")
        print("  streamlit run scripts/Deploy/web_demo.py")
        sys.exit(1)


_require_streamlit_run()

st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

WEB_CSS = """
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;  /* 改为圆形 */
            width: 32px !important;         /* 固定宽度 */
            height: 32px !important;        /* 固定高度 */
            padding: 0 !important;          /* 移除内边距 */
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;         /* 更柔和的颜色 */
            margin: 5px 10px 5px 0 !important;  /* 调整按钮间距 */
        }
        .stButton button:hover {
            border-color: #999 !important;
            color: #333 !important;
            background-color: #f5f5f5 !important;
        }
        .stMainBlockContainer > div:first-child {
            margin-top: -50px !important;
        }
        .stApp > div:last-child {
            margin-bottom: -35px !important;
        }
        
        /* 重置按钮基础样式 */
        .stButton > button {
            all: unset !important;  /* 重置所有默认样式 */
            box-sizing: border-box !important;
            border-radius: 50% !important;
            width: 18px !important;
            height: 18px !important;
            min-width: 18px !important;
            min-height: 18px !important;
            max-width: 18px !important;
            max-height: 18px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #888 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin: 0 2px !important;  /* 调整这里的 margin 值 */
        }

    </style>
"""
st.markdown(WEB_CSS, unsafe_allow_html=True)

system_prompt = []
device = "cuda" if torch.cuda.is_available() else "cpu"
image_url = "https://www.modelscope.cn/api/v1/studio/gongjy/MiniMind/repo?Revision=master&FilePath=images%2Flogo2.png&View=true"


@st.cache_resource
def load_model_tokenizer(cfg: ModelConfig):
    """按 ModelConfig 加载模型（与 chat_llm / serve_openai_api 共享 model_loader.init_model）。
    cfg.format='hf' → HF 格式目录；'native' → 原生 .pth 权重。"""
    return init_model(cfg)


def _user_bubble(content, live=False):
    """用户消息气泡 HTML（live=刚发送的高亮样式）。"""
    style = 'background-color: gray; color:white;' if live else 'background-color: #ddd; color: black;'
    return (f'<div style="display: flex; justify-content: flex-end;">'
            f'<div style="display: inline-block; margin: 10px 0; padding: 8px 12px 8px 12px; {style} border-radius: 10px;">'
            f'{content}</div></div>')


def render_sidebar():
    """侧边栏控件（保持 st.* 调用顺序）。返回运行配置 dict。"""
    st.sidebar.title("模型设定调整")
    st.session_state.history_chat_num = st.sidebar.slider("Number of Historical Dialogues", 0, 6, 0, step=2)
    st.session_state.max_new_tokens = st.sidebar.slider("Max Sequence Length", 256, 8192, 8192, step=1)
    st.session_state.temperature = st.sidebar.slider("Temperature", 0.6, 1.2, 0.85, step=0.01)
    st.session_state.enable_kv = st.sidebar.checkbox("启用跨轮 KV 缓存（多轮加速，保留完整历史）", value=False)

    model_source = st.sidebar.radio("选择模型来源", ["本地模型", "API"], index=0)

    if model_source == "API":
        api = {
            "url": st.sidebar.text_input("API URL", value="http://127.0.0.1:8000/v1"),
            "model_id": st.sidebar.text_input("Model ID", value="minimind"),
            "model_name": st.sidebar.text_input("Model Name", value="MiniMind2"),
            "key": st.sidebar.text_input("API Key", value="none", type="password"),
        }
        slogan = f"Hi, I'm {api['model_name']}"
        model_cfg, model_key, model_display = None, None, None
    else:
        model_format = st.sidebar.radio("模型格式", ["HF 格式", "原生 .pth"], index=0)
        if model_format == "原生 .pth":
            # 原生权重：resource/MiniMind2-PyTorch/<weight>_<hidden_size>[_moe].pth
            weight = st.sidebar.selectbox("权重阶段", NATIVE_WEIGHTS, index=1)  # 默认 full_sft
            hidden_size = st.sidebar.selectbox("hidden_size", [512, 640, 768], index=0)
            use_moe = st.sidebar.checkbox("MoE（640 自动开启）", value=(hidden_size == 640)) or hidden_size == 640
            model_cfg = ModelConfig(
                load_from='scripts/Model',
                save_dir='resource/MiniMind2-PyTorch',
                weight=weight,
                hidden_size=hidden_size,
                num_hidden_layers=16 if hidden_size == 768 else 8,
                use_moe=use_moe,
                device=device,
                format='native',
            )
            model_key = f"native:{weight}_{hidden_size}{'_moe' if use_moe else ''}"
            model_display = f"MiniMind-{weight}-{hidden_size}{'MoE' if use_moe else ''}"
        else:
            selected_model = st.sidebar.selectbox('Models', list(MODEL_PATHS.keys()), index=2)  # 默认选择 MiniMind2
            model_path = resolve_model_path(MODEL_PATHS[selected_model][0])
            model_display = MODEL_PATHS[selected_model][1]
            model_cfg = ModelConfig(load_from=model_path, device=device, format='hf')
            model_key = f"hf:{model_path}"
        slogan = f"Hi, I'm {model_display}"
        api = None

    return {
        "model_source": model_source,
        "api": api,
        "model_cfg": model_cfg,
        "model_key": model_key,
        "slogan": slogan,
        "show_thinking": (api is not None and 'R1' in api["model_name"]) or (api is None and 'R1' in model_display),
    }


def render_header(slogan):
    st.markdown(
        f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
        '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
        f'<img src="{image_url}" style="width: 45px; height: 45px; "> '
        f'<span style="font-size: 26px; margin-left: 10px;">{slogan}</span>'
        '</div>'
        '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容完全由AI生成，请务必仔细甄别<br>Content AI-generated, please discern with care</span>'
        '</div>',
        unsafe_allow_html=True
    )


def render_chat_history(messages, show_thinking):
    """渲染历史消息与删除按钮。"""
    for i, message in enumerate(messages):
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=image_url):
                st.markdown(process_assistant_content(message["content"], show_thinking), unsafe_allow_html=True)
                if st.button("×", key=f"delete_{i}"):
                    st.session_state.messages = st.session_state.messages[:i - 1]
                    st.session_state.chat_messages = st.session_state.chat_messages[:i - 1]
                    st.rerun()
        else:
            st.markdown(_user_bubble(message["content"]), unsafe_allow_html=True)


def _handle_api(cfg, placeholder):
    """调用 OpenAI 兼容 API 并流式渲染回复。"""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg["api"]["key"], base_url=cfg["api"]["url"])
        history_num = st.session_state.history_chat_num + 1  # +1 是为了包含当前的用户消息
        conversation_history = system_prompt + st.session_state.chat_messages[-history_num:]
        answer = ""
        for answer in api_generate(client, cfg["api"]["model_id"], conversation_history,
                                   st.session_state.temperature):
            placeholder.markdown(process_assistant_content(answer, cfg["show_thinking"]), unsafe_allow_html=True)
        return answer
    except Exception as e:
        answer = f"API调用出错: {str(e)}"
        placeholder.markdown(answer, unsafe_allow_html=True)
        return answer


def _handle_local(model, tokenizer, cfg, placeholder):
    """本地模型生成（支持跨轮前缀缓存），流式渲染回复。"""
    seed_generation()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    params = LocalParams(
        max_new_tokens=st.session_state.max_new_tokens,
        temperature=st.session_state.temperature,
        top_p=0.85,
        device=device,
        use_kv=st.session_state.enable_kv,
        hf_model=cfg["model_cfg"].format == 'hf',
    )

    if params.use_kv:
        messages_to_send = st.session_state.messages          # KV 模式用完整历史
    else:
        st.session_state.chat_messages = system_prompt + st.session_state.chat_messages[
                                                         -(st.session_state.history_chat_num + 1):]
        messages_to_send = st.session_state.chat_messages     # 原路径用窗口历史

    kv = {"cache": st.session_state.kv_cache, "all_ids": st.session_state.kv_all_ids}
    holder = local_generate(model, tokenizer, messages_to_send, params, kv, streamer)

    answer = ""
    for new_text in streamer:
        answer += new_text
        placeholder.markdown(process_assistant_content(answer, cfg["show_thinking"]), unsafe_allow_html=True)

    if params.use_kv:
        holder["thread"].join()  # 等生成线程把新 cache 写回 holder（streamer.end() 先于返回值赋值，需 join）
        st.session_state.kv_cache = holder.get("cache")
        st.session_state.kv_all_ids = kv["all_ids"]
    return answer


def handle_prompt(prompt, model, tokenizer, cfg):
    """处理一轮输入：渲染气泡、生成回复、记录消息。"""
    messages = st.session_state.messages
    st.markdown(_user_bubble(prompt, live=True), unsafe_allow_html=True)
    messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})
    st.session_state.chat_messages.append({"role": "user", "content": prompt[-st.session_state.max_new_tokens:]})

    with st.chat_message("assistant", avatar=image_url):
        placeholder = st.empty()
        if cfg["model_source"] == "API":
            answer = _handle_api(cfg, placeholder)
        else:
            answer = _handle_local(model, tokenizer, cfg, placeholder)

    messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    with st.empty():
        if st.button("×", key=f"delete_{len(messages) - 1}"):
            st.session_state.messages = st.session_state.messages[:-2]
            st.session_state.chat_messages = st.session_state.chat_messages[:-2]
            st.rerun()


def main():
    cfg = render_sidebar()
    render_header(cfg["slogan"])

    if cfg["model_source"] == "本地模型":
        model, tokenizer = load_model_tokenizer(cfg["model_cfg"])
    else:
        model, tokenizer = None, None

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_messages = []
    if "kv_cache" not in st.session_state:
        st.session_state.kv_cache = None
        st.session_state.kv_all_ids = None
        st.session_state.kv_model_path = None
    if cfg["model_source"] == "本地模型" and st.session_state.kv_model_path != cfg["model_key"]:
        # 切换了模型（含 HF/原生 互换）→ 旧缓存的 K/V 失效，重置
        st.session_state.kv_cache = None
        st.session_state.kv_all_ids = None
        st.session_state.kv_model_path = cfg["model_key"]

    messages = st.session_state.messages
    render_chat_history(messages, cfg["show_thinking"])

    prompt = st.chat_input(key="input", placeholder="给 MiniMind 发送消息")
    if hasattr(st.session_state, 'regenerate') and st.session_state.regenerate:
        prompt = st.session_state.last_user_message
        delattr(st.session_state, 'regenerate')
        delattr(st.session_state, 'last_user_message')
        delattr(st.session_state, 'regenerate_index')

    if prompt:
        handle_prompt(prompt, model, tokenizer, cfg)


if __name__ == "__main__":
    main()
