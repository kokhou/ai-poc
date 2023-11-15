from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.callbacks import get_openai_callback
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate

import urllib.parse
from dotenv import load_dotenv
import os
import tiktoken
import asyncio

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-instruct")

load_dotenv()


# def go_with_template():
#     TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
#     Use the following format:
#
#     Question: "Question here"
#     SQLQuery: "SQL Query to run"
#     SQLResult: "Result of the SQLQuery"
#     Answer: "Final answer here"
#
#     Only use the following tables:
#
#     {table_info}.
#
#     Some examples of SQL queries that correspond to questions are:
#
#     {few_shot_examples}
#
#     Question: {input}"""
#
#     CUSTOM_PROMPT = PromptTemplate(
#         input_variables=["input", "few_shot_examples", "table_info", "dialect"], template=TEMPLATE
#     )
#

# noinspection SqlDialectInspection
def normal_db_chain():
    db = SQLDatabase.from_uri("mysql+pymysql://root:" + urllib.parse.quote('P@ssw0rd') + "@localhost:3306/merchant")
    llm = OpenAI(temperature=0, verbose=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    # db_chain.run("How many employees are there?")

    TEMPLATE = """Given an input question, first create a syntactically correct mysql query to run, then look at the results of the query and return the answer.
    Use the following format:

    Question: "Question here"
    SQLQuery: "SQL Query to run"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"

    Only use the following tables:

    {table_info}.

    Some examples of SQL queries that correspond to questions are:

    {few_shot_examples}

    Question: {input}"""

    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["input", "few_shot_examples", "table_info"], template=TEMPLATE
    )
    CUSTOM_PROMPT.format(input="How much would it be like to hire a chef and a waiter?", few_shot_examples="""
 
    """, table_info="merchants")

    db_chain.run(CUSTOM_PROMPT)

# 1）我想在PJ开一家泰国餐厅，装修费为RM 500,000。针对中上市场，请问会成功吗？ - llm + predictive(can't do)
# 2) 如果请厨师和服务员大概工资多少？ - query database with chain
# 3) 我店铺大概有 1000方尺 ，需要请多少员工？ - llm + predictive(can't do)
# 4）平均要做多少生意才可以收支平衡？ - llm + predictive(can't do)
# 5）客户流量需要多少？平均客户单价需要多少？ llm + predictive(can't do)
# 6）这里附近的店铺租金是多少？ - query database with chain
# 7）开张我需要准备或注意什么事项？ - chatgpt + llm + predictive(can't do)
# 8）有什么促销能在的？ -
# 9）这3年有开泰国餐厅的倒闭吗？ - db
# 10）客户会注重什么细节？ - chatgpt + llm + predictive(can't do)
# 11) 开店3年生意无法突破有什么原因吗？chatgpt + llm + predictive(can't do)
# 12）我店最畅销产品是什么产品？ - db
# 13）我店面人流量最多是什么时候？ - db
# 14）什么季节我的生意最好 - db
# 15）我做过的促销那个最好生意？ - db
# 16）我的店现在最大问题是什么？
# 17）怎样增加收益？
# 18）我有能力开分店吗？
# 19）我的客户回流次数多吗？
# 20）有新客户平均多少？
# 21) 现在的客户多数是什么人？
# 22) Halal 和 Mesti 怎样申请？
# 23) 餐饮业会有什么挑战？
# 24) 哪一类餐饮目前最受欢迎？
# 25) 如何申请连锁执照？
# 26) 申请 Halal & Mesti 需要多少费用？
# 27) 申请 Halal & Mesti 要多久才能获得批准？
# 28) 申请连锁执照牵涉什么费用？
# 29) 经营餐饮业需要什么执照？
# 30) 员工的必须福利包括什么？

def go_database():
    db = SQLDatabase.from_uri("mysql+pymysql://root:" + urllib.parse.quote('P@ssw0rd') + "@localhost:3306/merchant")

    llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, verbose=True)

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
                
             """
             ),
            ("user", "{question}\n ai: "),
        ]
    ).format(
        question="what kind of questions I can ask this database"
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    with get_openai_callback() as cb:
        response = agent_executor.run(final_prompt)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    print(response)


def go_openai(message):
    print(f"This is my method. Value is {message}")


go_database()
# normal_db_chain()
