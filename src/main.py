import os

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


def document_based_answer(question: str, documents: list[dict]) -> str:
    with Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    ) as mistral:
        # retrieval
        # semantic embeddings
        # cfidf = inverted document frequency
        # similarity between question und doc#i, top3

        # 1000 documents, vector database, postgres, pgvector
        res = mistral.embeddings.create(model="mistral-embed", inputs=[question, *[doc["title"] for doc in documents]])

        similarities = []
        for element in res.data[1:]:
            similarities.append(
                np.dot(res.data[0].embedding, element.embedding)
                / (np.linalg.norm(res.data[0].embedding) * np.linalg.norm(element.embedding))
            )

        # map similarity to index

        print(similarities)
        # to give to the Gen
        suitable_document, best_similarity = max(zip(documents, similarities), key=lambda x: x[1])
        print(suitable_document["content"])

        # rerank
        # if best_similarity > 0.8:
        # Gen
        # isntead of system -> user mostly (depends on LLM)
        answer = mistral.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "content": "Instructions. Parse the following documents. After the documents, you will find a question. Parse the question and answer based on the documents only. If you cannot answer based on the documents, yield only 'Sorry.'",
                    "role": "system",
                },
                {"content": suitable_document["content"], "role": "system"},
                {
                    "content": question,
                    "role": "user",
                },
            ],
            stream=False,
        )

        # Handle response
        return answer.choices[0].message.content


if __name__ == "__main__":
    # system läuft, chunking, pgvector;

    question = "How to fry eggs?"

    # >1000 docs, embed all; report 500 Seiten
    SAMPLE_DOCUMENTS = [
        {
            "id": "doc_1",
            "title": "Python Programming Basics",
            "content": (
                "Python is a high-level programming language known for its simplicity and readability. "
                "It supports multiple programming paradigms including procedural, object-oriented, and functional programming."
            ),
        },
        {
            "id": "doc_2",
            "title": "Machine Learning Introduction",
            "content": (
                "Machine learning is a subset of artificial intelligence that enables computers to learn and improve "
                "from experience without being explicitly programmed. It uses algorithms to analyze data and make predictions."
            ),
        },
        {
            "id": "doc_3",
            "title": "Web Development Overview",
            "content": (
                "Web development involves building and maintaining websites. It includes aspects such as web design, web publishing, "
                "web programming, and database management. Common technologies include HTML, CSS, and JavaScript."
            ),
        },
        {
            "id": "doc_4",
            "title": "Data Structures Fundamentals",
            "content": (
                "Data structures are ways of organizing and storing data so that they can be accessed and worked with efficiently. "
                "Common data structures include arrays, linked lists, stacks, queues, trees, and graphs."
            ),
        },
        {
            "id": "doc_5",
            "title": "Introduction to Algorithms",
            "content": (
                "An algorithm is a step-by-step procedure to solve a problem or perform a computation. Algorithms are essential to computer science "
                "and are used for tasks such as searching, sorting, optimization, and machine learning."
            ),
        },
    ]

    answer = document_based_answer(question, SAMPLE_DOCUMENTS)
    print(answer)
