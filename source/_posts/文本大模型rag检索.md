---
title: 文本大模型rag检索
author: Miao HanCheng
date: 2025-04-28 15:00:00
tags:
  - deep learning
  - python
  - llm
  - rag
categories:
  - 技术文章
---



### 背景

在工作中会遇到有些文本的描写、语序、语种出现变幻，但是大体上意思是可以匹配上一些已知知识库的时候，原本的文本embedding向量检索是可以解决问题的，但是当语言含义需要结合现实知识理解后才能匹配，就需要用大模型先行理解，在做匹配



### 模型选取

目前是选用了 阿里开源的[qwq](https://huggingface.co/Qwen/QwQ-32B)，使用ollama快速部署

```shell
ollama run qwq
```

也对比过其他模型，比如：[DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)、[DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)、[Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)，还有几个别的模型，目前测下来还是qwq的效果最好



### 代码结构

1. 加载内部知识库，同时进行处理，处理完成后的内容通过 [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)来进行embedding，完成后导入本地FAISS，作为先验知识
2. 大模型先理解，解析数据后进行匹配，如果匹配上则直接返回输出
3. 如果步骤2匹配不到，大模型进行理解补全，规范化之后再进行匹配，匹配上则直接返回

为什么要加上3，因为现实生活中很多场景下实际上的输入是简称或者缩写，人类可以理解但是直接向量匹配有难度，需要大模型理解补全后进行匹配



### 代码逻辑

引入langchain、模型服务由ollama提供

```python
import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings, OllamaLLM  
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import traceback

```



配置模型后端服务，并测试

```python
# -----------------------------
# 配置 LLM（用于后处理重排序以及地址补全）
llm = OllamaLLM(
    model="qwq",   
    temperature=0.3,
    max_tokens=4096
)
try:
    response = llm.invoke('你好')
    print("LLM 返回：", response)
except Exception as e:
    print("LLM 调用失败，请检查模型配置：", e)

```



加载先验知识库（这里以一个地址信息为例，开源数据地址：[US-Cities-Database](https://github.com/kelvins/US-Cities-Database)）

```python
# -----------------------------
# 加载区划配置数据
column_names = ['REGION', 'COUNTY', 
             'CITY',  'LONGITUDE', 'LATITUDE','POST_CODE_LIST']
df = pd.read_csv("data.txt", sep="|", header=None, names=column_names)
print("原始配置数据预览：\n", df.head())
```

| CITY     | COUNTY         | LATITUDE  | LONGITUDE  |
| -------- | -------------- | --------- | ---------- |
| Adak     | Aleutians West | 55.999722 | -161.20777 |
| Akiachak | Bethel         | 60.891854 | -161.39233 |
| Akiak    | Bethel         | 60.890632 | -161.19932 |
| Akutan   | Aleutians East | 54.143012 | -165.78536 |
| Alakanuk | Kusilvak       | 62.746967 | -164.60228 |



构建文档，把pd转换成可用于检索的docs

```python

# -----------------------------
# 构建候选 Document 集合（针对所有数据，后续检索时再做过滤）
def build_candidate_documents(df_all):
    candidate_docs = []
    for _, row in df_all.iterrows():
        text = (
            f"国家: {row['REGION']} | 州: {row['COUNTY']} | "
            f"城市: {row['CITY']} | "
            f"邮编: {row['POST_CODE_LIST']}"
        )
        metadata = {
            "REGION": row['REGION'],
            "POST_CODE_LIST": row['POST_CODE_LIST']
        }
        candidate_docs.append(Document(page_content=text, metadata=metadata))
    return candidate_
```



初始化embedding模型

```python


# -----------------------------
# 初始化嵌入模型
try:
    embeddings = OllamaEmbeddings(
        model='nomic-embed-text', 
        base_url="http://localhost:11434",
        temperature=0.3
    )
except Exception as e:
    print("初始化 Embeddings 失败，请检查模型配置：", e)
    raise

```



FAISS索引（为了加快速度，我做了一个缓存，实际上可以每次都重新生成）

```python


# -----------------------------
# FAISS 索引缓存设置
INDEX_DIR = "faiss_index_dir"
if os.path.exists(INDEX_DIR):
    faiss_index = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print("加载了本地 FAISS 索引")
else:
    candidate_docs = build_candidate_documents(df)
    faiss_index = FAISS.from_documents(candidate_docs, embeddings)
    faiss_index.save_local(INDEX_DIR)
    print("构建并缓存了 FAISS 索引。")
```



定义好提示词模板和要求的结果格式：

```python

# -----------------------------
# 定义后处理重排序的提示模板
rerank_prompt_template = """
以下为候选区划匹配结果：
{context}

用户输入的地址为：{input}

请根据候选结果及用户输入，选择最匹配的一条（忽视大小写），并严格按照以下格式返回结果：
REGION: [国家]
COUNTY: [州]
CITY: [城市]
请仅返回匹配结果，不添加其他解释性文字。
"""
rerank_prompt = PromptTemplate(
    template=rerank_prompt_template,
    input_variables=["context", "input"]
)
document_chain = create_stuff_documents_chain(llm, rerank_prompt)

```



定义如果第一次匹配不到，补全的提示词和返回结果

```python

# -----------------------------
# 定义大模型补全地址的提示模板和函数
complete_address_template = """
用户输入的地址为：{input}
已知国家为：{region}
用户输入的有可能是快递中转点、机场、城市名、城市简写等，请你综合所有的知识来判断，补全地址的语种与用户输入的语种一致,州不需要加 state county 等描述
请将该地址补全为标准格式的城市，州，国家，格式如下：
REGION: [国家]
COUNTY: [州]
CITY: [城市]

请仅返回补全后的地址，不附加其他解释性文字。
"""
complete_address_prompt = PromptTemplate(
    template=complete_address_template,
    input_variables=["input", "region"]
)
```



解析结果

```python
import re

def parse_think_content(text):
    """解析包含think标签的文本并提取结构化地址"""
    # 提取think块中的answer内容（支持多段分析场景）
    answer_blocks = re.findall(r'</think>.*', text, re.DOTALL)
    if not answer_blocks:
        return None

    # 初始化地址字典
    address = {
        '城市': '',
        '州': '',
        '国家': ''
    }

    # 从最后一个answer块中提取结构化地址（优先取最终结论）
    last_answer = answer_blocks[-1].strip()
    for line in last_answer.split('\n'):
        if '城市' in line:
            address['城市'] = line.split(':')[-1].strip()
        elif '州' in line:
            address['州'] = line.split(':')[-1].strip()
        elif '国家' in line:
            address['国家'] = line.split(':')[-1].strip()

    # 拼接为指定格式
    return f"{address['城市']},{address['州']},{address['国家']}"


```





调用大模型补全

```python



def complete_address(address_input, region):
    prompt = complete_address_prompt.format(input=address_input, region=region)
    try:
        result = llm.invoke(prompt)

        return result.strip()
    except Exception as e:
        print("调用大模型补全地址时失败：", e)
        return address_input

```

检查doc对象

```python

# -----------------------------
# 辅助函数：确保候选项为 Document 对象
def ensure_document(doc):
    if isinstance(doc, Document):
        return doc
    else:
        return Document(page_content=str(doc), metadata={})
```

检索过滤

```python

# 检索后预过滤函数：基于用户输入的邮编和 region 过滤候选结果
def post_filter_candidates(user_input, user_region, candidates):
    filtered = candidates
    print(candidates)
    if user_region:
        filtered = [doc for doc in filtered if  doc.metadata.get("region", "").lower() == user_region.lower()]
    print(f"基于 region {user_region} 过滤后候选数：{len(filtered)}")
    postal_pattern = re.compile(r"[\s,;+]*([0-9]{5}(?:-[0-9]{4})?)")
    postal_match = postal_pattern.search(user_input)
    if postal_match:
        postal_code = postal_match.group(1)
        filtered = [doc for doc in filtered if postal_code.lower() in doc.metadata.get("post_code_list", "").lower()]
        print(f"用户输入中检测到邮编 {postal_code}，过滤后候选数：{len(filtered)}")

    return filtered

```



主要逻辑

```python

# -----------------------------
# 主交互模块
if __name__ == "__main__":
    print("=" * 40)
    print("智能地址区划匹配系统（全量向量化缓存+检索后预过滤+后处理重排序）")
    print("输入 'exit' 退出程序")
    print("=" * 40)

    retriever = faiss_index.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20, "score_threshold": 0.7},
        return_source_documents=True
    )

    while True:
        user_input = input("\n请输入待匹配的地址文本：").strip()
        user_input = user_input.replace('+', ',')
        if user_input.lower() == 'exit':
            break
        user_region = input("请输入对应的 region：").strip()

        print("正在定位...")

        try:
            # 第一次检索与过滤
            retrieved_docs = retriever.invoke(user_input)
            filtered_docs = post_filter_candidates(user_input, user_region, retrieved_docs)

            # 若没有匹配到任何候选结果，则尝试调用大模型补全地址，并重新检索
            if not filtered_docs:
                print("未匹配到任何数据，尝试通过大模型补全地址...")
                new_input = complete_address(user_input, user_region)
                print("大模型补全地址思考: ", new_input)
                new_input = parse_think_content(new_input)
                print("大模型补全后的地址：", new_input)
                retrieved_docs = retriever.invoke(new_input)
                filtered_docs = post_filter_candidates(new_input, user_region, retrieved_docs)
                if not filtered_docs:
                    print("补全地址后仍未匹配到任何区划数据，放弃该地址。")
                    continue
                else:
                    user_input = new_input

            # 根据候选数量决定是否调用 LLM 后处理重排序
            if len(filtered_docs) > 1:
                rerank_result = document_chain.invoke({
                    "context": filtered_docs,
                    "input": user_input
                })
                print("\n匹配结果：")
                print(rerank_result)
            else:
                print("\n匹配结果：")
                single_doc = filtered_docs[0]
                print(single_doc.page_content if hasattr(single_doc, "page_content") else str(single_doc))
        except Exception as e:
            traceback.print_exc()
            print(f"处理失败：{str(e)}")
```





### 运行效果

1. 直接就能匹配到：

```json
========================================
智能地址区划匹配系统（全量向量化缓存+检索后预过滤+后处理重排序）
输入 'exit' 退出程序
========================================

请输入待匹配的地址文本：33170,fl
请输入对应的 region：united states
正在定位...
基于 region united states 过滤后候选数：20
用户输入中检测到邮编 33170，过滤后候选数：1

匹配结果：
国家: United States | 州: Florida | 城市: Goulds 
```









2. 需要补全才能匹配上

```json
========================================
智能地址区划匹配系统（全量向量化缓存+检索后预过滤+后处理重排序）
输入 'exit' 退出程序
========================================

请输入待匹配的地址文本：lax
请输入对应的 region：United States
正在定位...
[]
基于 region United States 过滤后候选数：0
未匹配到任何区划数据，尝试通过大模型补全地址...
大模型补全地址思考:  <think>
好的，我现在需要处理用户输入的地址“lax”，并将其补全为城市、州和国家的标准格式。首先，我得确定用户可能指的是什么。

用户提到输入可能是快递中转点、机场、城市名或简写等。LAX 是一个常见的缩写，最著名的应该是洛杉矶国际机场（Los Angeles International Airport）。不过也有可能指其他地方，但通常LAX主要关联的是这个机场。接下来需要确认所属的行政区划。

洛杉矶国际机场位于美国加利福尼亚州洛杉矶市。因此，城市通常是具体的市或区。这里应该填洛杉矶市（City of Los Angeles）。州是加州（California），一级地址自然是United States。用户要求州不需要添加state、county等描述，所以直接写California即可。

需要确保所有名称都是完整的正式名称，并且语种一致，用户输入的是英文缩写，所以补全后的地址也应使用英文。检查是否有其他可能性，比如是否存在以LAX命名的其他城市或地区，但考虑到国际通用性，LAX作为机场代码更为常见，因此确定上述结构是正确的。

</think>

城市: Los Angeles
州: California
国家: United States
大模型补全后的地址： Los Angeles,California,United States

基于 lv1_name United States 过滤后候选数：20

匹配结果：
<think>
好的，我需要根据用户提供的地址“Los Angeles, California, United States”从候选区划中找到最匹配的一条。首先，用户输入的是国家United States，州是California，城市是Los Angeles。不过候选中的城市有多个包含“Los Angeles”的选项，比如Lake Los Angeles、Los Angeles Afb、West Los Angeles、East Los Angeles等。

我需要仔细看一下每个候选条目。用户直接写的是Los Angeles，没有更具体的区域，所以可能要找最接近的匹配。注意到有一个城市是“Los Angeles Afb”，邮编90009。但可能存在更准确的选项吗？

不过候选列表中并没有一个完全精确匹配“Los Angeles”作为城市的情况，因为通常洛杉矶市（City of Los Angeles）可能没有被单独列为一个条目，而是分为不同的区域如West、East等。这时候需要判断哪个最接近用户输入。

或者可能用户输入的地址是通用的，而候选中的选项更具体。例如，“Los Angeles Afb”可能指的是洛杉矶空军基地，但用户可能指整个城市。不过根据提供的候选列表中没有完全匹配“Los Angeles”的城市名称，所以必须选择最接近的。

比较各个选项，其中“Los Angeles Afb”虽然带有Afb（可能是空军基地），但城市名称中最接近用户输入的是它，因为其他如Lake Los Angeles和East/West Los Angeles都是更具体的区域。因此可能选Los Angeles Afb作为匹配？

或者是否有可能我遗漏了某个条目？再仔细检查一遍候选列表：

候选中的城市有：
- Lake Los Angeles
- Los Angeles Afb
- West Los Angeles
- East Los Angeles
- Hollywood（属于LA市的一部分）
- Beverly Hills（虽然属于洛杉矶县，但州是CA）

用户输入的是Los Angeles作为城市，而可能正确的应该是整个城市名称。但候选中没有直接的“Los Angeles”条目，所以需要选最接近的。可能用户指的是整个城市，但候选中的选项更细分了区域。这时候可能需要选择邮编覆盖市中心或主要区域的那个？

例如，“Los Angeles Afb”的邮编是90009，而洛杉矶市中心的一些邮编可能属于其他区域。或者West Los Angeles的邮编包括90028等。不过用户输入的是最通用的形式，没有具体到东西湖或空军基地，所以可能需要选择最接近名称匹配的选项，即“Los Angeles Afb”？

但另一个可能性是，虽然名称不完全相同，但用户可能指的是整个洛杉矶市，而候选中可能将该市分为不同区域。因此，在这种情况下，可能应该选择邮编覆盖主要区域的那个条目？或者根据名称相似度，最接近的是“Los Angeles Afb”，因为其他选项如Lake Los Angeles明显是更小的社区。

不过也有可能用户输入的地址中的城市其实是洛杉矶县（County），但候选中的州已经是California，所以城市应该是城市或地区。可能在这种情况下，“Los Angeles Afb”是最直接匹配名称的部分，因此选它？

或者是否还有其他考虑因素？比如邮编是否覆盖更广泛的区域？例如，如果用户没有提供具体地址部分，那么最合适的可能是最通用的选项。但候选中并没有这样的条目。

综上所述，在提供的候选列表中最接近用户输入“Los Angeles”的城市名称是“Los Angeles Afb”，因此选择该条目。
</think>

国家: United States
州: California
城市: Los Angeles Afb
```



3. 从候选中思考后选择最匹配的（无需补全）

```json
========================================
智能地址区划匹配系统（全量向量化缓存+检索后预过滤+后处理重排序）
输入 'exit' 退出程序
========================================

请输入待匹配的地址文本：north miami beach ,fl
请输入对应的 region：united states
正在定位...
基于 region united states 过滤后候选数：7

匹配结果：
<think>
好的，我需要处理用户的查询，他们输入的地址是“north miami beach, fl”，并从给定的候选数据中找到最匹配的一条。首先，我要仔细分析用户提供的每个候选信息。

首先，用户输入中的关键词是“North Miami Beach”和“FL”。根据问题要求，要忽视大小写，所以需要将所有选项转换为统一的大小写形式进行比较。接下来，我会逐一查看各个候选区划：

1. **North Miami Beach**：城市名称正好与用户的输入完全匹配，并且州简称是FL，符合用户提供的州缩写。其邮编包括33181、33179等，可能覆盖用户提到的地区。

2. **North Miami**：虽然名字中有“North Miami”，但缺少“Beach”部分，与用户的输入不完全一致。不过需要确认是否存在拼写错误或简称的情况，但根据问题要求应严格匹配名称。

3. 其他选项如North Palm Beach、North Redington Beach、Miami Beach等，名称差异较大，明显不符合用户输入的“North Miami Beach”。


此外，用户的输入中有逗号分隔城市和州，通常格式为“City, State”。这里用户明确写了FL（佛罗里达州），与候选中的州一致。

综上所述，最匹配的应该是第一个选项：North Miami Beach。
</think>

国家: United States
州: Florida
城市: North Miami Beach

```



理解简写后匹配

```json
========================================
智能地址区划匹配系统（全量向量化缓存+检索后预过滤+后处理重排序）
输入 'exit' 退出程序
========================================

请输入待匹配的地址文本：jfk,ny,us
请输入对应的 region：united states
正在定位...
基于 lv1_name united states 过滤后候选数：20

匹配结果：
<think>
嗯，用户输入的地址是“jfk, ny, us”。我需要从候选区划中找到最匹配的一条。首先，国家应该是United States，因为用户用了US，而所有候选项都是国家为United States。

接下来州是NY，对应New York州。现在要看城市部分。用户输入的JFK应该是指约翰·菲茨杰拉德·肯尼迪国际机场，通常缩写为JFK。在候选列表里有几个可能的选项：

第一个候选条目就是John F Kennedy Airpor，邮编11430。另一个是Kennedy，邮编14747。这两个名字都和JFK相关，但哪一个更准确呢？

通常来说，纽约的肯尼迪机场正式名称是John F. Kennedy International Airport，所以全称应该是John F Kennedy Airport。而另一个Kennedy可能是指其他地方，比如纽约州内的某个镇或区域？不过邮编14747查一下是否属于该机场附近？或者可能Kennedy是简称？

用户输入的是jfk，这通常直接对应到John F Kennedy Airport这个全称。所以应该选第一个条目。

其他选项比如Roosevelt Island、Loehmanns Plaza这些显然不符合JFK的缩写。而Florida的那个Kennedy Space Center虽然也有Kennedy，但州是Florida，用户明确用了NY，所以排除。

因此最匹配的是第一个条目。
</think>

国家: United States
州: New York
城市: John F Kennedy Airport
```

