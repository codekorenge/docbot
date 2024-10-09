import configparser




def __read_config():
    pass


class Configuration:
    def __init__(self):
        config = configparser.ConfigParser()

        # Read the configuration file
        config.read('config.ini')

        # Access values from the configuration file
        # debug_mode = config.getboolean('General', 'debug')
        app_name = config.get('App', 'name')
        app_data = config.get('App', 'data')
        app_metadata = config.get('App', 'metadata')
        app_progress = config.getboolean('App', 'progress')
        app_verbose = config.getboolean('App', 'verbose')
        app_prompts = config.get('App', 'prompts')

        embed_name = config.get('Embed', 'name')

        llm_name = config.get('LLM', 'name')
        llm_token_limit = config.getint('LLM', 'token_limit')
        llm_temperature = config.getfloat('LLM', 'temperature')

        chunk_size = config.getint('Chunk', 'size')
        chunk_overlap = config.getint('Chunk', 'overlap')

        ret_max = config.getint('Retriever', 'max')
        ret_score = config.getfloat('Retriever', 'score')

        # db_name = config.get('Database', 'db_name')
        # db_host = config.get('Database', 'db_host')
        # db_port = config.get('Database', 'db_port')

        # Return a dictionary with the retrieved values
        self.config_values = {
            'app_name': app_name,
            'app_data': app_data,
            'app_metadata': app_metadata,
            'app_progress': app_progress,
            'app_verbose': app_verbose,
            'app_prompts': app_prompts,
            'embed_name': embed_name,
            'llm_name': llm_name,
            'llm_token_limit': llm_token_limit,
            'llm_temperature': llm_temperature,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'ret_max': ret_max,
            'ret_score': ret_score,

            # 'log_level': log_level,
            # 'db_name': db_name,
            # 'db_host': db_host,
            # 'db_port': db_port
        }

#         return config_values
#
#
#         self.config_values =
# # @staticmethod
# def __read_config():
#     config = configparser.ConfigParser()
#
#     # Read the configuration file
#     config.read('config.ini')
#
#     # Access values from the configuration file
#     # debug_mode = config.getboolean('General', 'debug')
#     app_name = config.get('Application', 'app_name')
#     # db_name = config.get('Database', 'db_name')
#     # db_host = config.get('Database', 'db_host')
#     # db_port = config.get('Database', 'db_port')
#
#     # Return a dictionary with the retrieved values
#     config_values = {
#         'debug_mode': app_name,
#         # 'log_level': log_level,
#         # 'db_name': db_name,
#         # 'db_host': db_host,
#         # 'db_port': db_port
#     }
#
#     return config_values