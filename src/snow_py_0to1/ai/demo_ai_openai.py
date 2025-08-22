from openai import OpenAI
import os
import json

# 设置OpenAI API密钥 (建议使用环境变量管理密钥)
# 在终端执行: export OPENAI_API_KEY='your-api-key'
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY 环境变量未设置。请设置您的API密钥。")

# 创建客户端实例
client = OpenAI(api_key=api_key)


def ask_openai(prompt, model="gpt-3.5-turbo", max_tokens=2000):
    """
    调用OpenAI聊天API并获取响应
    参数:
        prompt: 用户输入的提示文本
        model: 使用的模型 (默认: gpt-3.5-turbo)
        max_tokens: 生成的最大token数量 (默认: 500)
    返回:
        API的完整响应对象
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,  # 控制随机性 (0.0-2.0)
            top_p=0.9,  # 控制多样性
            n=1,  # 返回的结果数量
        )
        return response
    except Exception as e:
        print(f"API调用错误: {e}")
        return None


def parse_response(response):
    """
    解析OpenAI API的响应
    参数:
        response: API返回的响应对象
    返回:
        dict: 包含解析结果的字典
    """
    if not response:
        return None

    # 提取基本信息
    result = {
        "id": response.id,
        "model": response.model,
        "created": response.created,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        },
        "choices": []
    }

    # 解析每个选择
    for choice in response.choices:
        choice_data = {
            "index": choice.index,
            "finish_reason": choice.finish_reason,
            "message": {
                "role": choice.message.role,
                "content": choice.message.content,
                # 如果有工具调用，可以在这里添加
            }
        }
        result["choices"].append(choice_data)

    return result


if __name__ == "__main__":
    # 示例调用
    user_prompt = "用简单的语言解释量子计算的基本原理"

    # 调用API
    api_response = ask_openai(user_prompt)

    if api_response:
        # 解析响应
        parsed = parse_response(api_response)

        # 打印解析结果
        print(f"\n==== 完整响应解析 ====")
        print(f"模型: {parsed['model']}")
        print(f"ID: {parsed['id']}")
        print(f"创建时间: {parsed['created']}")
        print(
            f"使用情况: {parsed['usage']['prompt_tokens']} 提示词token + {parsed['usage']['completion_tokens']} 完成token = {parsed['usage']['total_tokens']} total")

        # 打印主要回复内容
        print("\n==== AI回复内容 ====")
        for choice in parsed["choices"]:
            print(f"\n[回复 {choice['index'] + 1}]")
            print(choice["message"]["content"])

        # 可选：将完整响应保存为JSON文件
        with open("openai_response.json", "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print("\n完整响应已保存到 openai_response.json")
    else:
        print("API调用失败")