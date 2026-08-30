set dotenv-load

default_input_backend := env_var_or_default('AOE2_INPUT_BACKEND', 'auto')

install:
    uv sync

agent *ARGS:
    uv run --package gameplay-agent aoe2-agent {{ARGS}}

agent-lmstudio *ARGS:
    AOE2_LLM_WIRE=lmstudio AOE2_INPUT_BACKEND="{{default_input_backend}}" \
    uv run --package gameplay-agent aoe2-agent {{ARGS}}

agent-linux input="auto" *ARGS:
    AOE2_INPUT_BACKEND={{input}} \
    uv run --package gameplay-agent aoe2-agent {{ARGS}}

test-linux:
    AOE2_INPUT_BACKEND=silent uv run python -m pytest \
        tests/test_lmstudio_wire.py \
        tests/test_input_backend_linux.py \
        -q
