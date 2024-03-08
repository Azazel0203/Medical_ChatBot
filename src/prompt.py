prompt_template="""
Use the following pieces of information and the chat history (delimited by <hs></hs>) to answer the user's question.
If you don't know the answer, simply state that you don't know; avoid making up an answer.

Question: {question}
Information: {context}

<hs>
{history}
</hs>

Helpful Answer:
"""
