## 1. 写在前面
借助llama-index库来实践基于LLM的RAG应用
ref: https://docs.llamaindex.ai/en/stable/index.html


## 2. 基于gemini和bge来完成llama_index的Starter Tutorial
ref: https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html

**LLM API选择**

llama_index默认使用openAI api，对于快速上手来说最合适的就是openai api。
* 私有部署小模型：没有GPU资源
* 购买openai API配额：信用卡死活不通过
* 最终选择Google gemini pro：llama_index支持gemini调用（本来以为不支持，配置openai proxy也可以通过模仿openai api来调用）

**embedding模型选择**

llama_index默认调用openAI的embedding模型，但是可以选择本地模型（实际是调用HuggingFaceEmbedding），默认是```BAAI/bge-small-en```

```python
# llm选择和embedding模型选择
service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')
```
logging的debug日志可以看到与huggingface仓库的交互：
```
DEBUG:urllib3.connectionpool:Starting new HTTPS connection (1): huggingface.co:443
Starting new HTTPS connection (1): huggingface.co:443
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/config.json HTTP/1.1" 200 0
DEBUG:urllib3.connectionpool:https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
https://huggingface.co:443 "HEAD /BAAI/bge-small-en/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
```

**加载本地文档**

```Python
documents = SimpleDirectoryReader(data_dir).load_data()
```
日志：
```
DEBUG:llama_index.readers.file.base:> [SimpleDirectoryReader] Total files added: 1
> [SimpleDirectoryReader] Total files added: 1
```

**对本地文档做chunk操作**

```python
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
```
Vector Stores是retrieval-augmented generation(RAG)的关键组成部分。当调用from_documents，文件会被分割为chunks并解析为Node objects,这是对文本字符串的轻量化抽象，来保持元数据和关系。

这种方式默认在内存中存储vector embedding，稍后会存下来方便后续加载（节省时间和embedding调用花销）。

日志：
```
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: What I Worked On

February 2021

Before college...
> Adding chunk: What I Worked On

February 2021

Before college...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: I couldn't have put this into words when I was ...
> Adding chunk: I couldn't have put this into words when I was ...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Over the next several years I wrote lots of ess...
> Adding chunk: Over the next several years I wrote lots of ess...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: So we just made what seemed like the obvious ch...
> Adding chunk: So we just made what seemed like the obvious ch...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: I don't think it was entirely luck that the fir...
> Adding chunk: I don't think it was entirely luck that the fir...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: YC was different from other kinds of work I've ...
> Adding chunk: YC was different from other kinds of work I've ...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: For the rest of 2013 I left running YC more and...
> Adding chunk: For the rest of 2013 I left running YC more and...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Now they are, though. Now you could continue us...
> Adding chunk: Now they are, though. Now you could continue us...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Notes

[1] My experience skipped a step in the ...
> Adding chunk: Notes

[1] My experience skipped a step in the ...
DEBUG:llama_index.node_parser.node_utils:> Adding chunk: Customary VC practice had once, like the custom...
> Adding chunk: Customary VC practice had once, like the custom...
```

**基于index实例化query_engine**

```python
query_engine = index.as_query_engine(service_context=service_context)
```
方法定义如下：
```
    def as_query_engine(self, **kwargs: Any) -> BaseQueryEngine:
        # NOTE: lazy import
        from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

        retriever = self.as_retriever(**kwargs)

        kwargs["retriever"] = retriever
        if "service_context" not in kwargs:
            kwargs["service_context"] = self._service_context
        return RetrieverQueryEngine.from_args(**kwargs)
```
其中的as_retriever方法实例化了一个VectorIndexRetriever对象：
```python
class VectorIndexRetriever(BaseRetriever):
    """Vector index retriever.

    Args:
        index (VectorStoreIndex): vector store index.
        similarity_top_k (int): number of top k results to return.
        vector_store_query_mode (str): vector store query mode
            See reference for VectorStoreQueryMode for full list of supported modes.
        filters (Optional[MetadataFilters]): metadata filters, defaults to None
        alpha (float): weight for sparse/dense retrieval, only used for
            hybrid query mode.
        doc_ids (Optional[List[str]]): list of documents to constrain search.
        vector_store_kwargs (dict): Additional vector store specific kwargs to pass
            through to the vector store at query time.

    """
```

**query执行**

```python
response = query_engine.query("What did the author do growing up?")
print('Answer: ', response)
```
调用的方法：
```python
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._guideline_evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryGuidelineQueryEngine(
                self._query_engine,
                self._guideline_evaluator,
                self.resynthesize_query,
                self.max_retries - 1,
                self.callback_manager,
            )
            new_query = self.query_transformer.run(query_bundle, {"evaluation": eval})
            logger.debug("New query: %s", new_query.query_str)
            return new_query_engine.query(new_query)
```

日志：
```
DEBUG:llama_index.indices.utils:> Top 2 nodes:
> [Node 473dffaf-d857-4bc2-8a01-6b0d92112ba7] [Similarity score:             0.806073] What I Worked On

February 2021

Before college the two main things I worked on, outside of schoo...
> [Node dd7262a5-e922-48dc-8907-48641e27f1c9] [Similarity score:             0.784516] I couldn't have put this into words when I was 18. All I knew at the time was that I kept taking ...
> Top 2 nodes:
> [Node 473dffaf-d857-4bc2-8a01-6b0d92112ba7] [Similarity score:             0.806073] What I Worked On

February 2021

Before college the two main things I worked on, outside of schoo...
> [Node dd7262a5-e922-48dc-8907-48641e27f1c9] [Similarity score:             0.784516] I couldn't have put this into words when I was 18. All I knew at the time was that I kept taking ...
```

最终输出答案：
```
Answer: Growing up, the author worked on writing and programming. He wrote short stories and tried writing programs on an IBM 1401 computer. Later, he got a microcomputer and started programming more seriously, writing simple games, a program to predict how high his model rockets would fly, and a word processor that his father used to write a book.
```


完整代码：
```python
import os
os.environ["GOOGLE_API_KEY"] = "xxx-xxx-xxx"
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import Gemini

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # 比较详细的logging等级，方便学习
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

data_dir = '../data'

service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')

documents = SimpleDirectoryReader(data_dir).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)
response = query_engine.query("What did the author do growing up?")
print('Answer: ', response)
```

**存储index内容**

只需一行代码：
```python
index.storage_context.persist(persist_dir = '../data')
```

可以看到存储到了制定文件夹:
```bash
» ls -lh ../data
total 824
-rw-r--r--@ 1 jade_mayer  staff   187K  1  1 23:41 default__vector_store.json
-rw-r--r--@ 1 jade_mayer  staff   135K  1  1 23:41 docstore.json
-rw-r--r--@ 1 jade_mayer  staff    18B  1  1 23:41 graph_store.json
-rw-r--r--@ 1 jade_mayer  staff    72B  1  1 23:41 image__vector_store.json
-rw-r--r--@ 1 jade_mayer  staff   2.0K  1  1 23:41 index_store.json
-rw-r--r--@ 1 jade_mayer  staff    73K  1  1 22:18 paul_graham_essay.txt
```
embedding向量在default__vector_store.json里面，可以看到每个token的输出都是384维，正是```BAAI/bge-small-en```的输出维度

ref： https://huggingface.co/BAAI/bge-small-en

```
{
    "embedding_dict": {
        "32ba1f7f-3ef5-4118-b3ce-172b4f127168": [
            -0.04698172211647034,
            -0.015481043606996536,
            0.023415159434080124,
            -0.03800944611430168,
            0.018414050340652466,
            -0.0031388888601213694,
            -0.006215846631675959,
            0.029209135100245476,
            -0.010553239844739437,
            -0.014835268259048462,
            0.034739766269922256,
            0.034783363342285156,
            0.027504893019795418,
            -0.022163057699799538,
            -0.0014071145560592413,
            -0.004258439876139164,
            -0.0198955275118351,
            -0.01113078836351633,
            -0.017337610945105553,
            0.030663305893540382,
            0.007644901517778635,
            -0.03388264775276184,
            -0.018255650997161865,
            -0.03617030754685402,
            0.030288150534033775,
            0.04922771826386452,
            -0.02133074402809143,
            -0.02667289972305298,
            -0.016942422837018967,
            -0.20316243171691895,
            -0.012253538705408573,
            -0.016811028122901917,
            0.04966513812541962,
            0.004801961127668619,
            0.013124674558639526,
            -0.000572275253944099,
            0.031204206869006157,
            0.005266610532999039,
            -0.03495708480477333,
            -0.0032691622618585825,
            -0.008665332570672035,
            -0.0069218585267663,
            -0.03163096681237221,
            -0.0038232721854001284,
            0.017784595489501953,
            -0.015017405152320862,
            -0.02884853258728981,
            -0.035970624536275864,
            0.04844813793897629,
            0.007230944931507111,
            -0.00853420328348875,
            -0.016010424122214317,
            -0.026202332228422165,
            0.008808224461972713,
            -0.004676107782870531,
            -0.003882945980876684,
            0.05168820917606354,
            0.050272274762392044,
            0.004477598704397678,
            -0.025546498596668243,
            0.005372005980461836,
            0.013710727915167809,
            -0.19433408975601196,
            0.06069575622677803,
            -0.0019249554025009274,
            0.01481388509273529,
            -0.03563850745558739,
            -0.041716307401657104,
            0.026342442259192467,
            0.05208372697234154,
            -0.026825036853551865,
            0.013046554289758205,
            -0.020447321236133575,
            0.05694744363427162,
            0.013076210394501686,
            -0.016684945672750473,
            -0.010218190029263496,
            -0.01419403962790966,
            0.030298857018351555,
            -0.0006505550118163228,
            -0.012502582743763924,
            -0.033863216638565063,
            0.005387178622186184,
            0.004914384335279465,
            -0.07887214422225952,
            0.01294825877994299,
            -0.0025863356422632933,
            0.012086167000234127,
            0.009021352045238018,
            -0.036562222987413406,
            -0.0048615047708153725,
            -0.00794854573905468,
            0.024151042103767395,
            0.006930611561983824,
            -0.043348103761672974,
            -0.04225297272205353,
            0.03534917160868645,
            0.02466086857020855,
            0.003643989795818925,
            0.6611256003379822,
            -0.027740545570850372,
            -0.014811787754297256,
            0.04347195848822594,
            -0.010510500520467758,
            0.017281142994761467,
            -0.006882404442876577,
            0.020195890218019485,
            -0.05597776547074318,
            -0.024992171674966812,
            -0.0032624949235469103,
            0.02329074591398239,
            -0.02709822915494442,
            0.024455759674310684,
            -0.013449798338115215,
            0.06372255086898804,
            -0.0038755363784730434,
            0.010372409597039223,
            0.013081301003694534,
            -0.019358709454536438,
            0.010305331088602543,
            -0.0502275675535202,
            0.055949851870536804,
            -0.028020024299621582,
            -0.027101553976535797,
            -0.026879573240876198,
            -0.08787120878696442,
            0.025118974968791008,
            0.05361277237534523,
            0.01179643627256155,
            -0.005580545868724585,
            0.02967955730855465,
            -0.012361369095742702,
            -0.05280441418290138,
            -0.0009390964405611157,
            0.015578185208141804,
            0.010047201067209244,
            0.014594684354960918,
            -0.027308709919452667,
            0.05666743591427803,
            -0.01745647005736828,
            -0.014582964591681957,
            0.0012075741542503238,
            -0.0005419948138296604,
            -0.010487139225006104,
            0.0021557658910751343,
            0.06975462287664413,
            0.02567513845860958,
            0.010253284126520157,
            -0.008600698783993721,
            0.022716019302606583,
            0.005362640600651503,
            -0.0017097460804507136,
            -0.031566865742206573,
            -0.014000901021063328,
            0.020275287330150604,
            0.017717117443680763,
            0.06214563548564911,
            0.0016782801831141114,
            -0.028010167181491852,
            0.0076273647136986256,
            -0.035920776426792145,
            0.0233931764960289,
            -0.03981495276093483,
            0.04712972417473793,
            0.05012526363134384,
            -0.06567484140396118,
            0.0052355010993778706,
            0.03285547345876694,
            0.022370390594005585,
            -0.043199148029088974,
            0.0576508566737175,
            -0.008379493840038776,
            0.010484708473086357,
            -0.014891906641423702,
            0.011855938471853733,
            -0.0028156512416899204,
            -0.036108050495386124,
            0.0011416403576731682,
            0.03309705853462219,
            0.014114928431808949,
            0.01676088571548462,
            -0.008011307567358017,
            0.008729408495128155,
            -0.014578762464225292,
            -0.015731068328022957,
            -0.04351452738046646,
            -0.026935799047350883,
            0.03289668262004852,
            0.0073287165723741055,
            0.023215817287564278,
            -0.03248995169997215,
            0.028128135949373245,
            -0.06486396491527557,
            -0.01092046033591032,
            -0.031219322234392166,
            -0.0014282348565757275,
            0.04481466859579086,
            0.0015789347235113382,
            -0.014495582319796085,
            -0.058139629662036896,
            -0.010771508328616619,
            -0.0010938802734017372,
            0.008928176946938038,
            0.04303095117211342,
            -0.004482574295252562,
            -0.0043173134326934814,
            0.022726159542798996,
            -0.0113997096195817,
            0.04317191615700722,
            0.004497899208217859,
            -0.003584994236007333,
            -0.04186040163040161,
            0.024481210857629776,
            -0.008107219822704792,
            -0.053471341729164124,
            -0.0008622985333204269,
            0.008605959825217724,
            0.029924333095550537,
            0.006441369187086821,
            0.007594279479235411,
            0.005335251800715923,
            -0.0503290593624115,
            -0.043398357927799225,
            -0.23509953916072845,
            -0.006085492670536041,
            -0.006885690614581108,
            -0.0021820301190018654,
            0.014917004853487015,
            -0.0477629117667675,
            -0.01072191447019577,
            -0.029510077089071274,
            -0.010405691340565681,
            0.04056863486766815,
            0.026849722489714622,
            -0.01381123811006546,
            -0.015194986946880817,
            -0.04233722388744354,
            -0.00288687483407557,
            -0.000024224827939178795,
            0.006294459570199251,
            -0.01755005307495594,
            -0.03659006580710411,
            -0.0006270630401559174,
            -0.011915943585336208,
            -0.030001897364854813,
            -0.06073204427957535,
            -0.037339210510253906,
            0.024144046008586884,
            -0.013735479675233364,
            0.15554000437259674,
            0.0365920215845108,
            0.0898180902004242,
            -0.006376015488058329,
            0.010851170867681503,
            0.01847461611032486,
            0.012753809802234173,
            -0.09338387846946716,
            0.006150288973003626,
            0.025476796552538872,
            0.006668054964393377,
            0.008094497956335545,
            -0.02185792848467827,
            -0.01799268275499344,
            -0.03570615127682686,
            0.017315296456217766,
            -0.009562759660184383,
            -0.0666576400399208,
            -0.06772714853286743,
            -0.02395460568368435,
            -0.047934722155332565,
            -0.04223375394940376,
            0.0039051298517733812,
            0.016795765608549118,
            0.0013074687449261546,
            -0.016004296019673347,
            -0.0042895362712442875,
            0.00542974378913641,
            -0.009387962520122528,
            -0.020903877913951874,
            -0.08615951240062714,
            0.07688149064779282,
            0.02149123139679432,
            0.022567955777049065,
            -0.010352427139878273,
            -0.008106427267193794,
            -0.0019004123751074076,
            -0.009731575846672058,
            0.020128842443227768,
            0.03239506110548973,
            0.003256630850955844,
            -0.029615595936775208,
            0.012562848627567291,
            0.010300596244633198,
            0.002722091507166624,
            0.06664852052927017,
            -0.008797467686235905,
            -0.0011255150893703103,
            0.05247491970658302,
            -0.011824097484350204,
            0.017473319545388222,
            0.015303699299693108,
            0.004609926603734493,
            -0.0576062873005867,
            0.011326410807669163,
            -0.049580760300159454,
            0.008977160789072514,
            0.013591749593615532,
            0.008825071156024933,
            0.020945020020008087,
            0.06253630667924881,
            -0.0029495067428797483,
            0.03294045850634575,
            0.012859470210969448,
            -0.029512953013181686,
            -0.016141915693879128,
            -0.003895787987858057,
            0.020141346380114555,
            0.017351225018501282,
            -0.005995817482471466,
            -0.25368309020996094,
            0.003131302073597908,
            -0.006585310213267803,
            0.0009366677259095013,
            0.028124261647462845,
            0.0028949398547410965,
            0.011889223009347916,
            0.008783569559454918,
            -0.019197838380932808,
            0.018775703385472298,
            0.056931570172309875,
            0.018770283088088036,
            0.025767464190721512,
            0.021793469786643982,
            -0.005633120890706778,
            0.028209663927555084,
            0.050566039979457855,
            0.007110361475497484,
            0.01610880345106125,
            0.051287174224853516,
            0.025082813575863838,
            0.01705036871135235,
            0.16765068471431732,
            -0.01856471784412861,
            -0.0073897442780435085,
            -0.025082280859351158,
            0.00982898473739624,
            0.012578483670949936,
            0.025539681315422058,
            0.014613193459808826,
            0.015036532655358315,
            -0.02035905048251152,
            0.012077131308615208,
            -0.030746152624487877,
            0.027192451059818268,
            -0.01725425198674202,
            0.006972603499889374,
            0.05780875310301781,
            0.01598987728357315,
            0.006660557352006435,
            -0.026917191222310066,
            -0.0028073545545339584,
            -0.014330701902508736,
            -0.03930317237973213,
            0.02456594631075859,
            0.03348737210035324,
            -0.010120414197444916,
            -0.04479987919330597,
            -0.006718606222420931,
            -0.01666620559990406,
            -0.02210931107401848,
            0.010297814384102821,
            -0.0043894085101783276,
            0.007319907657802105,
            0.001180644379928708,
            0.05359862372279167,
            -0.03344108536839485,
            -0.024397986009716988,
            -0.008710783906280994,
            -0.008376377634704113,
            0.02223694697022438,
            -0.018686581403017044,
            0.016676589846611023,
            0.04485808312892914,
            0.02181057073175907
        ],
        // ...
    },
    "text_id_to_ref_doc_id": {
        "32ba1f7f-3ef5-4118-b3ce-172b4f127168": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "65d0c3f3-b2a9-4bb4-91be-ad42987529b5": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "5c235a2d-7113-48fd-bde0-13b8753a0d1f": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "a849b1ad-8a96-4f69-8eae-0c956e775db3": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "ce623617-0fa8-46f8-8fe1-a3cf89dc566b": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "51115e89-43e0-40eb-892d-c98476a29bbc": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "717dca2c-189d-430e-90d6-f6d866ab115f": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "d47de045-85f9-4a8f-adcc-e0ffb787d501": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "f8322cd8-9642-404b-9445-27829ced38d4": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "b8ff915a-428a-47c2-98be-ee7ace3b2d1c": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "4dd1e910-7ce9-4853-a0fa-8dd8c6cadeb5": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "b46df6c5-c1a5-48f0-bbf9-de33879fa87c": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "dbad6cfd-7c2f-4116-93c5-22260c1383ba": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "effff458-bfac-47d5-be8c-3e40f2d24f54": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "448432d1-4bb9-437a-9f46-db92b4d5ed70": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "fba535cb-0028-4d1b-ab55-59fff2f197e6": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "c5fdb8b0-9d91-4acd-a24f-90aa9d459ea4": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "6a7a062b-7086-4e45-a5c2-d95e8e1e28f9": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "973527a1-ceac-4c5a-b9e8-b17f842c3d44": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "48b7e978-dbb8-4764-9ce8-91979e03e8a8": "93d53b99-545d-466f-a1e0-50a62ab22951",
        "063cbd8a-22a3-4d44-93d0-a8509005f5bf": "93d53b99-545d-466f-a1e0-50a62ab22951"
    },
    "metadata_dict": {
        "32ba1f7f-3ef5-4118-b3ce-172b4f127168": {
            "file_path": "../data/paul_graham_essay.txt",
            "file_name": "paul_graham_essay.txt",
            "file_type": "text/plain",
            "file_size": 75042,
            "creation_date": "2024-01-01",
            "last_modified_date": "2024-01-01",
            "last_accessed_date": "2024-01-01",
            "_node_type": "TextNode",
            "document_id": "93d53b99-545d-466f-a1e0-50a62ab22951",
            "doc_id": "93d53b99-545d-466f-a1e0-50a62ab22951",
            "ref_doc_id": "93d53b99-545d-466f-a1e0-50a62ab22951"
        },
        // ...
    }
}
```

直接加载存储好的文档embedding来重复这个任务：
```python
import os
os.environ["GOOGLE_API_KEY"] = "xxx-xxx-xxx"
# from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms import Gemini

# import logging
# import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG) # 比较详细的logging等级，方便学习
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# data_dir = '../data'

# service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')

# documents = SimpleDirectoryReader(data_dir).load_data()
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# query_engine = index.as_query_engine(service_context=service_context)
# response = query_engine.query("What did the author do growing up?")
# print('Answer: ', response)

# index.storage_context.persist(persist_dir = '/Users/jade_mayer/projects/agents/llamaindex/data')

import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
service_context = ServiceContext.from_defaults(llm=Gemini(), embed_model='local')
# check if storage already exists
PERSIST_DIR = "../data"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("../data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context, service_context=service_context)

# either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```
