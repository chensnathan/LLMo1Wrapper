class o1Wrapper:
    def __init__(self, llm_client, api_key, model=None):
        if model is not None:
            self.llm_client = llm_client(api_key, model=model)
        else:
            self.llm_client = llm_client(api_key)
        
        # system prompts
        self.thinking_model_system_prompt = "你是一位思维助理，能够根据已有的思考过程，提供下一步的思考方向。请仔细阅读用户提供的内容，给出合理的下一步建议。注意只需要给出思考方向的标题就可以，不需要其他的内容。"
        self.answering_model_system_prompt = "你是一位专业的分析师，能够根据当前的思考方向和之前的讨论内容，提供深入的分析和结论。请仔细阅读用户提供的内容，给出详细的见解和建议。"
        self.reflection_model_system_prompt = "你是一位严谨的反思助手，擅长分析和修正回答内容。请仔细检查提供的答案，若存在不合理之处，请修正并提供完善的答案；若没有问题，直接返回原答案即可。"
        self.judgment_model_system_prompt = "你是一位智能判断助手，能够根据以上的思考内容，评估是否已经足以回答用户的问题。请只回答“是”或“否”，不需要其他解释。"
        self.final_answer_model_system_prompt = "你是一位专业的完成回复撰写员，能够根据用户的问题和提供的思考过程，生成详细的回答。请将你的回答以 Markdown 格式呈现。"

        # max_think_step
        self.max_think_step = 20

    def thinking_model(self, messages):
        thinking_messages = messages + [{"role": "user", "content": "请给出要回答用户问题的下一步的思考方向。"}]
        next_direction_generator = self.llm_client.chat(
            thinking_messages, self.thinking_model_system_prompt, stream=False)
        return "".join([direction for direction in next_direction_generator])

    def answering_model(self, current_direction, messages):
        answering_messages = messages + [{"role": "user", "content": f"针对当前的思考方向：“{current_direction}”，请给出详细的分析和结论。"}]
        answer_generator = self.llm_client.chat(
            answering_messages, self.answering_model_system_prompt, stream=False)
        return "".join([answer for answer in answer_generator])
    
    def reflection_model(self, current_answer):
        reflection_messages = [{"role": "user", "content": f"当前的回答是：“{current_answer}”，请分析现在答案的内容是否有不合理的地方，如果有，修复答案中不合理的地方。"}]
        reflection_generator = self.llm_client.chat(
            reflection_messages, self.reflection_model_system_prompt, stream=False)
        return "".join([reflection_answer for reflection_answer in reflection_generator])

    def judgment_model(self, user_query, thinking_process):
        markdown_thinking = ""
        for step in thinking_process:
            content = step["content"]
            reflection = step["reflection"]
            cur_content = "\n> ".join(content.split("\n"))
            cur_reflection = "\n> ".join(reflection.split("\n"))
            markdown_thinking += f"> **{step['direction']}**\n>\n> {cur_content}\n\n**反思**\n>\n> {cur_reflection}"

        judgment_message = [{"role": "user", "content": f"用户问题：{user_query}\n当前思维链以及结果：{markdown_thinking}\n你的判断："}]
        judgment_generator = self.llm_client.chat(
            judgment_message, self.judgment_model_system_prompt, stream=False)
        
        judgment = ""
        for judge in judgment_generator:
            judgment += judge
        if "是" in judgment:
            return 1, markdown_thinking
        elif "否" in judgment:
            return 0, markdown_thinking
        else:
            return 0, markdown_thinking  # 默认继续思考

    def final_answer_model(self, user_query, markdown_thinking):
        # 生成最终答案
        final_answer_prompt = (
            f"用户的问题是：“{user_query}”。\n\n"
            f"基于以下思考过程，请给出一个完整的、直接可用的答案，"
            f"并以Markdown格式的报告形式呈现。\n\n"
            f"{markdown_thinking}"
        )
        final_answer_messages = [{"role": "user", "content": final_answer_prompt}]

        final_answer_generator = self.llm_client.chat(
            final_answer_messages, self.final_answer_model_system_prompt, stream=True)
        # 返回包含思考过程和最终答案的完整Markdown内容
        return final_answer_generator

    def o1_response(self, user_query):
        messages = [{"role": "user", "content": user_query}]
        thinking_process = []

        cur_step = 0

        while cur_step <= self.max_think_step:
            # Step 1: 思考模型，获取下一步思考方向
            current_direction = self.thinking_model(messages)
            messages.append({"role": "assistant", "content": current_direction})

            # Step 2: 回答模型，获取当前思考方向的结果 + 验证答案
            current_content = self.answering_model(current_direction, messages)
            messages.append({"role": "assistant", "content": current_content})
            reflection = self.reflection_model(current_content)
            messages.append({"role": "assistant", "content": reflection})

            cur_step += 1

            # 记录思考过程
            thinking_process.append({
                "direction": current_direction,
                "content": current_content,
                "reflection": reflection
            })

            # Step 3: 判断模型，是否可以生成最终答案
            judge_flag, markdown_thinking = self.judgment_model(user_query, thinking_process)
            # Yield the updated thinking process
            yield {"type": "thinking", "content": markdown_thinking}
            
            if judge_flag:
                break  # 可以生成最终答案
            else:
                continue  # 继续思考下一步

        # Step 4: 最终回答模型，生成完整答案
        final_answer_generator = self.final_answer_model(user_query, markdown_thinking)
        for chunk in final_answer_generator:
            # Yield the final answer chunks
            yield {"type": "final_answer", "content": chunk}
