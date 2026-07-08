FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl gcc ca-certificates && \
    update-ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Source packages — import chain: algorithm → unfiltering → standard_method → tp7/srfs
COPY pyproject.toml .
COPY unfiltered_radiances/ ./unfiltered_radiances/
COPY prod/ ./prod/
COPY tp7/ ./tp7/
COPY srfs/ ./srfs/
# SRF CSVs — tp7.py resolves _DEFAULT_SRF_DIR relative to its own __file__
COPY data/SRF/ ./data/SRF/
# Coefficient file — auto-discovered from coefficients/ at runtime via _find_coefficient_file()
COPY coefficients/ ./coefficients/

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="$PATH:/root/.local/bin"
RUN poetry lock && poetry sync --only main --no-root

# Make local packages importable without editable install
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "unfiltered_radiances/algorithm.py"]
CMD [""]