import os
import platform
import edge_tts
from playsound import playsound
import asyncio
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("E:\workspace\Pythonprojects\model\chatglm2-6b-int4",
                                          trust_remote_code=True)
model = AutoModel.from_pretrained("E:\workspace\Pythonprojects\model\chatglm2-6b-int4",
                                  trust_remote_code=True).cuda()
# 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
# from utils import load_model_on_gpus
# model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    past_key_values, history = None, []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    times = 0
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        current_length = 0
        allText = ""
        for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history,
                                                                    past_key_values=past_key_values,
                                                                    return_past_key_values=True):
            if stop_stream:
                stop_stream = False
                break
            else:
                txt = response[current_length:]
                allText += txt
                # print(txt, end='', flush=True)
                current_length = len(response)
        print(allText)
        voice = 'zh-CN-YunxiNeural'
        output = './sound/chat_times_{}.mp3'.format(times)
        rate = '-4%'
        volume = '+0%'
        asyncio.run(my_function(allText,voice,rate,volume,output))
        # tts = edge_tts.Communicate(text=allText, voice=voice, rate=rate, volume=volume)
        # tts.save(output)
        playsound(output)

        times = times + 1


async def my_function(text: str, voice: str, rate: str, volume: str, output: str):
    tts = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    await tts.save(output)


if __name__ == "__main__":
    main()
