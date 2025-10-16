from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import os

# Initialize model and embeddings
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.1  # Low temperature for factual accuracy
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

# Sample company documentation
SAMPLE_DOCS = [
    """
    Remote Work Policy - Updated 2024

    Eligibility: All full-time employees who have completed 90 days of employment
    are eligible for remote work arrangements.

    Requirements:
    - Stable internet connection (minimum 50 Mbps)
    - Dedicated workspace with professional background for video calls
    - Available during core hours: 10 AM - 3 PM local time
    - Response time: Within 2 hours during working hours

    Equipment: Company provides laptop, monitor, and $500 home office stipend.
    Employees are responsible for internet costs.

    Frequency: Up to 3 days per week for hybrid roles, 5 days for fully remote positions.
    Managers have discretion to adjust based on team needs.
    """,
    """
    Expense Reimbursement Policy

    Eligible Expenses:
    - Travel: Flights (economy class), hotels (up to $200/night), meals (up to $50/day)
    - Equipment: Software licenses, peripherals, books
    - Training: Courses, conferences, certifications

    Submission Process:
    1. Submit expenses within 30 days using Expensify app
    2. Include itemized receipts for all expenses over $25
    3. Provide business justification for expenses over $500
    4. Manager approval required before reimbursement

    Reimbursement Timeline: 7-10 business days after approval

    Non-Reimbursable: Alcohol, personal entertainment, luxury upgrades,
    traffic violations, personal errands during business trips.
    """,
    """
    PTO (Paid Time Off) Policy

    Accrual Rates:
    - Years 0-2: 15 days per year (1.25 days per month)
    - Years 3-5: 20 days per year (1.67 days per month)
    - Years 6+: 25 days per year (2.08 days per month)

    Additional:
    - 10 company holidays
    - 5 sick days (separate from PTO)
    - Unlimited sick days for serious illness with doctor's note

    Request Process:
    - Submit requests at least 2 weeks in advance
    - Holiday periods require 1 month advance notice
    - Manager approval needed
    - Max 2 consecutive weeks without VP approval

    Rollover: Up to 5 unused days can roll over to next year
    Payout: Unused PTO is paid out at 50% upon termination
    """
]

print("✓ Environment setup complete")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap to maintain context
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Process documents
chunks = []
for i, doc in enumerate(SAMPLE_DOCS):
    doc_chunks = text_splitter.split_text(doc)
    # Add metadata for source tracking
    chunks.extend([
        {"content": chunk, "source": f"document_{i+1}"}
        for chunk in doc_chunks
    ])

print(f"✓ Created {len(chunks)} chunks from {len(SAMPLE_DOCS)} documents")
print(f"Sample chunk: {chunks[0]['content'][:100]}...")

# Create vector database
texts = [chunk["content"] for chunk in chunks]
metadatas = [{"source": chunk["source"]} for chunk in chunks]

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    collection_name="company_docs"
)

print("✓ Vector store created")

# Test similarity search
test_query = "How many days of PTO do I get?"
results = vectorstore.similarity_search(test_query, k=2)

print(f"\n=== Testing Retrieval ===")
print(f"Query: {test_query}")
print(f"\nTop result:\n{results[0].page_content[:200]}...")

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
)

# Create QA prompt template
qa_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful HR assistant who answers employee questions
    based on company policies.

    Guidelines:
    - Only use information from the provided context
    - If information is not in the context, say "I don't have that information"
    - Cite specific policy sections when possible
    - Be concise but complete
    - Use friendly, professional tone"""),
    ("human", """Context from company documents:
    {context}

    Employee Question: {question}

    Answer:""")
])

# Build chain with LCEL
def format_docs(docs):
    """Format retrieved documents into context string."""
    return "\n\n".join([f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
                        for doc in docs])

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | qa_template
    | model
    | StrOutputParser()
)

print("✓ QA Chain built successfully")

# Test questions
test_questions = [
    "What equipment does the company provide for remote work?",
    "How long do I need to work here before I can work remotely?",
    "What's the maximum hotel cost per night for business travel?",
    "How much PTO do I get after 4 years?",
    "Can I get reimbursed for alcohol at a business dinner?",
    "What happens to my unused PTO if I leave the company?"
]

print("\n" + "="*60)
print("QUESTION ANSWERING TEST")
print("="*60)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*60}")
    print(f"Q{i}: {question}")
    print("="*60)

    # Get answer
    answer = qa_chain.invoke(question)
    print(f"A: {answer}")

    # Show retrieved context (for debugging)
    retrieved_docs = retriever.get_relevant_documents(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant chunks")
    print(f"Top source: {retrieved_docs[0].metadata['source']}")
