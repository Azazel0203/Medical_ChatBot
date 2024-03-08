prompt_template="""
Use the following pieces of information and the chat history to answer the user's question.
If you don't know the answer, simply state that you don't know; avoid making up an answer.
Keep the answer short and crisp

Question: {question}
Information: {context}
History: {history}

If there is nothing in the history, Please ignore it.


Helpful Answer:
"""
