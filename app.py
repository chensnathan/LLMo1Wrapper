import streamlit as st

from llms import GLM4LLM, Qwen2LLM
from llm_o1_wrapper import o1Wrapper


def main():
    st.title("OpenAI-o1-reproducing")

    # 在侧边栏输入 API 密钥
    api_key = st.sidebar.text_input("请输入您的 API_KEY：", type="password")

    # 模型选项
    model_options = ["glm-4-flash", "Qwen/Qwen2-7B-Instruct", ""]
    model_choice = st.sidebar.selectbox("请选择模型名称：", model_options)
    model = model_choice
    if "glm" in model:
        llm_client = GLM4LLM
    elif "Qwen" in model:
        llm_client = Qwen2LLM
    else:
        llm_client = GLM4LLM

    if api_key:
        # 创建 o1Wrapper 实例
        llm_o1 = o1Wrapper(llm_client, api_key, model=model)

        # 输入用户问题
        user_query = st.text_input("请输入内容：")

        if st.button("提交"):
            # 直接展示用户输入的内容
            st.write("您的输入是：", user_query)

            if user_query.strip() == "":
                st.warning("请输入内容。")
            else:
                # 显示生成响应的进度
                with st.spinner("正在生成响应..."):
                    # 调用 o1_response 方法并收集输出
                    response_generator = llm_o1.o1_response(user_query)

                    # 初始化占位符
                    thinking_expander = st.expander("显示思考过程")
                    thinking_placeholder = thinking_expander.empty()
                    final_answer_placeholder = st.empty()

                    # 初始化变量
                    markdown_thinking = ""
                    final_answer = "## 最终答案\n\n"

                    # 处理响应生成器
                    for item in response_generator:
                        if item['type'] == 'thinking':
                            # 更新思考过程
                            markdown_thinking = item['content']
                            thinking_placeholder.markdown(markdown_thinking)
                        elif item['type'] == 'final_answer':
                            # 更新最终答案
                            final_answer += item['content']
                            final_answer_placeholder.markdown(final_answer)
    else:
        st.warning("请在侧边栏输入您的 API_KEY。")

if __name__ == "__main__":
    main()