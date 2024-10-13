from llama_index.core.chat_engine.types import  AgentChatResponse

from helpers.app_config import Configuration


def display_retrieval_metrics(response:AgentChatResponse, config: Configuration):
    print(f"Configured params: max-results: {config.config_values['ret_max']}, "
          f"min-score: {config.config_values['ret_score']}.\r\n")
    print("Retrieved chunk(s):")
    cnt = 0
    for node in response.source_nodes:
        cnt += 1
        print(f"[{cnt}], "
              f"\tdoc-name:{node.metadata['file_name']}, "
              f"\tscore:{node.score}, "           
              f"\ttext:{node.text[0:30]} ...")
