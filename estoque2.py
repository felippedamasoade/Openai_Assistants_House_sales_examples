import pandas as pd
import openpyxl
import openai
from dotenv import load_dotenv
import os
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Carregar variáveis de ambiente
load_dotenv()

# Configurar chave da API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


def carregar_csv(file_path):
    """
    Carrega o arquivo CSV e retorna o DataFrame.
    
    Parâmetros:
        file_path (str): Caminho do arquivo CSV.
        
    Retorno:
        DataFrame: Dados do CSV.
    """
    return pd.read_csv(file_path, sep=';')


def criar_assistente(file_path):
    """
    Cria o assistente e faz o upload do arquivo.
    
    Parâmetros:
        file_path (str): Caminho do arquivo CSV.
        
    Retorno:
        dict: Informações do assistente criado.
    """
    with open(file_path, 'rb') as f:
        file = openai.files.create(file=f, purpose='assistants')

    assistant = openai.beta.assistants.create(
        name='Analista de dados em uma empresa de varejo',
        instructions='Fornecer e analisar dados e gerar relatórios no qual possa trazer métricas e fornecer respostas '
                     'do que se trata a planilha. Especificamente é uma planilha de estoque de uma empresa de varejo que está em .csv',
        tools=[{'type': 'code_interpreter'}],
        tool_resources={'code_interpreter': {'file_ids': [file.id]}},
        model='gpt-4o'
    )

    thread = openai.beta.threads.create()
    return {'assistant': assistant, 'file': file, 'thread': thread}


def enviar_mensagem(thread_id, texto_mensagem):
    """
    Envia uma mensagem para o assistente.
    
    Parâmetros:
        thread_id (str): ID da thread.
        texto_mensagem (str): Mensagem a ser enviada.
        
    Retorno:
        dict: Mensagem criada.
    """
    return openai.beta.threads.messages.create(
        thread_id=thread_id,
        role='user',
        content=texto_mensagem
    )


def executar_assistente(thread_id, assistant_id):
    """
    Executa o assistente e retorna o resultado.
    
    Parâmetros:
        thread_id (str): ID da thread.
        assistant_id (str): ID do assistente.
        
    Retorno:
        dict: Resultado da execução.
    """
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions='O nome do usuário é Felippe e ele é premium.'
    )

    # Aguardar a conclusão da execução
    while run.status in ['queued', 'in_progress', 'cancelling']:
        time.sleep(1)
        run = openai.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id
        )
    return run


def processar_resultados(thread_id, run):
    """
    Processa e exibe os resultados do assistente.
    """
    try:
        if run.status == 'completed':
            print("Execução concluída com sucesso. Processando resultados...")
            run_steps = openai.beta.threads.runs.list(thread_id=thread_id)

            for step in run_steps.data[::-1]:
                if hasattr(step, 'step_details') and step.step_details:
                    print(f'\n=== Tipo de Passo: {step.step_details.type}')
                    if step.step_details.type == 'message_creation':
                        message = openai.beta.threads.messages.retrieve(
                            thread_id=thread_id,
                            message_id=step.step_details.message_creation.message_id
                        )
                        if hasattr(message, 'content') and message.content:
                            for content_item in message.content:
                                if content_item.type == 'text':
                                    print("Mensagem do Assistente:")
                                    print(content_item.text.value)
                                elif content_item.type == 'image_file':
                                    file_id = content_item.image_file.file_id
                                    try:
                                        print("Baixando gráfico gerado...")
                                        image_data = openai.files.download(file_id)
                                        # Salvar e exibir a imagem
                                        file_path = f'{file_id}.png'
                                        with open(file_path, 'wb') as f:
                                            f.write(image_data)
                                        img = mpimg.imread(file_path)
                                        plt.imshow(img)
                                        plt.axis('off')
                                        plt.show()
                                    except Exception as img_error:
                                        print(f"Erro ao baixar/exibir a imagem: {img_error}")
        else:
            print(f"Erro durante a execução. Status: {run.status}")
    except Exception as e:
        print(f"Erro ao processar os resultados: {e}")

def exibir_graficos(thread_id, run):
    """
    Exibe os gráficos gerados pelo assistente.
    """
    try:
        print("Exibindo gráficos gerados...")
        run_steps = openai.beta.threads.runs.list(thread_id=thread_id)

        for step in run_steps.data[::-1]:
            print(f"Verificando passo: {step}")
            if hasattr(step, 'step_details') and step.step_details:
                if step.step_details.type == 'message_creation':
                    message = openai.beta.threads.messages.retrieve(
                        thread_id=thread_id,
                        message_id=step.step_details.message_creation.message_id
                    )
                    if hasattr(message, 'content') and message.content:
                        for content_item in message.content:
                            if content_item.type == 'image_file':
                                print(f"Baixando gráfico: {content_item.image_file.file_id}")
                                file_id = content_item.image_file.file_id
                                image_data = openai.files.download(file_id)

                                # Salvar e exibir a imagem
                                file_path = f'{file_id}.png'
                                with open(file_path, 'wb') as f:
                                    f.write(image_data)
                                print(f"Gráfico salvo em: {file_path}")
                                img = mpimg.imread(file_path)
                                plt.imshow(img)
                                plt.axis('off')
                                plt.show()
    except Exception as e:
        print(f"Erro ao exibir gráficos: {e}")



def main():
    """
    Função principal que controla o fluxo do programa.
    """
    try:
        # Carregar o arquivo CSV
        file_path = input("Digite o caminho do arquivo CSV: ")
        df = carregar_csv(file_path)
        print("Dados carregados com sucesso:")
        print(df.head())

        # Criar assistente
        assistente_info = criar_assistente(file_path)
        assistant_id = assistente_info['assistant'].id
        thread_id = assistente_info['thread'].id

        # Entrada e saída do usuário
        texto_mensagem = input("\nDigite sua mensagem para o assistente: ")
        enviar_mensagem(thread_id, texto_mensagem)
        
        # Executar o assistente
        run = executar_assistente(thread_id, assistant_id)

        # Processar os resultados
        processar_resultados(thread_id, run)
        exibir_graficos(thread_id, run)

    except Exception as e:
        print(f"Ocorreu um erro na execução do programa: {e}")


if __name__ == "__main__":
    main()
