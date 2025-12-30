FROM python:3.12-slim

LABEL org.opencontainers.image.title="SmartQL"
LABEL org.opencontainers.image.description="Natural Language to SQL - Database First"
LABEL org.opencontainers.image.url="https://github.com/smartql/smartql"
LABEL org.opencontainers.image.source="https://github.com/smartql/smartql"
LABEL org.opencontainers.image.vendor="SmartQL"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install uv for fast installs
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e ".[server]"

# Create non-root user
RUN useradd -m -u 1000 smartql
USER smartql

EXPOSE 8000

# Default command runs the HTTP server
CMD ["smartql", "serve", "--host", "0.0.0.0", "--port", "8000"]
