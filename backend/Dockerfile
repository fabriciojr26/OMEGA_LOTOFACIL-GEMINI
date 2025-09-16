FROM python:3.11

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências
COPY backend/requirements.txt .

# Instala as dependências. A imagem base completa garante as ferramentas de build.
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY backend/ .

# Expõe a porta e define o host
ENV PORT=8080
ENV HOST=0.0.0.0

# Comando para iniciar o servidor
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

