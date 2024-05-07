import os
import tempfile
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from config import WHITE, GREEN, RESET_COLOR, model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from questions import ask_question, QuestionContext

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                print("No documents were found to index. Exiting.")
                exit()
            print("Repository cloned. Indexing files...")

            llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.2)

            template = """
            You are an assistant tasked with answering questions about a codebase.

            The context:
            - Repo: {repo_name}
            - GitHub URL: {github_url}
            - Numbered Documents: {numbered_documents}
            - File Count: {file_type_counts}
            - File Names: {filenames}

            Question: {question}

            Answer clearly:
            1. Analyze the codebase to understand relevant details.
            2. Provide direct answers based on the code context.
            3. If unsure, admit that and refrain from guessing.
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=[
                    "repo_name",
                    "github_url",
                    "numbered_documents",
                    "question",
                    "file_type_counts",
                    "filenames"
                ]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)
            conversation_history = ""
            question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
            while True:
                try:
                    user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == "exit()":
                        break
                    print('Thinking...')
                    user_question = format_user_question(user_question)
                    answer = ask_question(user_question, question_context)
                    print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                    conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
            print("Failed to clone the repository.")
