

## 笔记

原始 ipynb文件 [bonus-unit1.ipynb - Colab](https://colab.research.google.com/#fileId=https%3A//huggingface.co/agents-course/notebooks/blob/main/bonus-unit1/bonus-unit1.ipynb&scrollTo=e63f4962-c644-491e-aa91-50e453e953a4)

1. 关于数据集若干问题
    1. 这里的原始数据集 只有一个split，就是train。 即类型是 DatasetDict，只有一个key="train"。
    2. 数据下载
        1. 数据已经存在本地缓存，所以就算是执行是抛出网络错误，也不会影响加载数据集。 但会非常耗时，需要很多的网络尝试。
        2. 也可以通过设置环境变量 HF_DATASETS_OFFLINE=1 来避免网络请求。
    3. datasetDict.map 是作用于 value的dataset类型的。
    4. **NousResearch/hermes-function-calling-v1 是指原始数据，不包含think过程。本项目中加载的数据是基于该数据集创建的新数据集，是加入了deepseek的思考过程的。**
    5. 从加载的数据中，可以看出说句中有的时候开启了思考，有的时候没有开启思考，说明不同任务有不同的难度，这种数据特性使得模型也可以学习到这种能力。
2. 训练
    1. gemma2模型在T4GPU中无法训练，所以使用了 gemma3-1b-it 模型，使得可以在T4 GPU中运行。并且两者的 special_token尽量兼容。
      1. 那么不兼容会怎样？ 比如使用 llama类型的chat训练数据来微调gemma模型，是否会出问题？
    2. 警告信息：`The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 1}. `
        1. 其实是说 原始的模型原始的tokenizer中的special_token和  SFTTrainer中对于训练数据处理使用的 tokenizer 是不同的。
        2. 模型 `model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation='eager', device_map="auto")` 中使用的是原始模型文件中定义的tokenizer。
        3. SFTTrainer 中加入了 `<tools> <think> <tool_call> <tool_response>`
        4. 但我在模型原始的tokenizer中找不到 start_of_turn end_of_turn两个特殊的token。
3. 训练数据的消息格式非常相似，但使用该模板考虑了对训练数据中，去除了不合理的消息。

新的消息格式
```Python
{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '
' + message['content'] | trim 
+ '<end_of_turn> ' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model '}}{% endif %}
```

模型原有的消息格式(gemma2)。Gemma3的消息模型更复杂，不过基本结构相同。
```Python
{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim 
+ '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}
```



## 训练数据结构说明

数据来源 https://huggingface.co/datasets/Jofthomas/hermes-function-calling-thinking-V1

```json
[
  {
    "content": "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:
    <tools> 
    [
        {
        'type': 'function',
        'function': {
            'name': 'get_stock_price',
            'description': 'Get the current stock price of a company',
            'parameters': {
                'type': 'object',
                'properties': {
                    'company': {
                        'type': 'string',
                        'description': 'The name of the company'
                        }
                    },
                'required': ['company']
                }
            }
        },
        {
        'type': 'function',
        'function': {
            'name': 'get_movie_details',
            'description': 'Get details about a movie',
            'parameters': {
                'type': 'object',
                'properties': {
                    'title': {
                        'type': 'string',
                        'description': 'The title of the movie'
                        }
                    },
                'required': ['title']
                }
            }
        }
    ]
    </tools>
    Use the following pydantic model json schema for each tool call you will make: 
    {'title': 'FunctionCall', 'type': 'object', 
        'properties': 
            { 
            'arguments': {'title': 'Arguments', 'type': 'object'}, 
            'name': {'title': 'Name', 'type': 'string'}
            }, 
        'required': ['arguments', 'name']
    }
    For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:\n<tool_call>\n{tool_call}\n</tool_call>",
    "role": "system"
  },
  {
    "content": "Hi, can you tell me the current stock price of Apple?",
    "role": "human"
  },
  {
    "content": "<think>Alright, the user asked for the current stock price of Apple. Looking at the tools available, there's a function called get_stock_price which exactly fits this request. I'll need to call this function with the parameter company set to 'Apple' to retrieve the information the user is asking for. This seems like the most straightforward and appropriate action to take.\n</think>
    <tool_call>\n{'name': 'get_stock_price', 'arguments': {'company': 'Apple'}}\n</tool_call>",
    "role": "model"
  },
  {
    "content": "<tool_response>\n{'stock_price': '$150.75'}\n</tool_response>",
    "role": "tool"
  },
  {
    "content": "The current stock price of Apple is $150.75.",
    "role": "model"
  },
  {
    "content": "That's great. Can you also tell me the stock price for Microsoft?",
    "role": "human"
  },
  {
    "content": "<think>Alright, let me try to figure out how to handle this user's request. The user first asked about Apple's stock price, and I responded by calling the get_stock_price function with 'Apple' as the argument. The tool provided the price, and I passed that information back to the user.\n\nNow, the user is asking for the stock price of Microsoft. I need to determine the appropriate action here. Looking at the tools available, there's still the get_stock_price function, which is designed to fetch current stock prices. Since the user is requesting another company's stock price, it makes sense to use the same function again but with the company name changed to Microsoft. \n\nI shouldn't assume any other tools are needed because the available functions are specifically tailored for stock prices and movie details. In this case, the relevant function is get_stock_price. I should ensure that the function is called with the correct parameter, 'Microsoft' in this case, to retrieve the accurate information the user is seeking. This approach is efficient and aligns directly with the user's request without making unnecessary assumptions or using unrelated tools.\n</think>
    <tool_call>\n{'name': 'get_stock_price', 'arguments': {'company': 'Microsoft'}}\n</tool_call>",
    "role": "model"
  },
  {
    "content": "<tool_response>\n{'stock_price': '$210.22'}\n</tool_response>",
    "role": "tool"
  },
  {
    "content": "The current stock price of Microsoft is $210.22.",
    "role": "model"
  },
  {
    "content": "Thank you for the information.",
    "role": "human"
  },
  {
    "content": "You're welcome! If you have any other questions, feel free to ask.",
    "role": "model"
  }
]
```