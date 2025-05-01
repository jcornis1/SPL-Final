
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def run_test(code, env=None):
    if env is None:
        env = {}
    result, _ = evaluate(parse(tokenize(code)), env)
    return result, env

def test_apikey_and_mode():
    print("Running API Key and Mode tests...")

    # Define API keys
    code = 'apikey openai = "sk-abc"; apikey weather = "wx-123";'
    result, env = run_test(code)
    assert env["openai"]["value"] == "sk-abc"
    assert env["weather"]["value"] == "wx-123"

    # Define modes
    code = 'mode debug uses openai; mode forecast uses weather;'
    _, env = run_test(code, env)
    assert env["__modes__"]["debug"] == "openai"
    assert env["__modes__"]["forecast"] == "weather"

    # Set current mode and simulate call
    code = 'currentMode = "debug"; callMode("Check my code");'
    run_test(code, env)  # No assertion here, but output should simulate call

    # Switch mode and call again
    code = 'currentMode = "forecast"; callMode("What’s the weather?");'
    run_test(code, env)

    print("✅ All mode and API key tests passed!")

if __name__ == "__main__":
    test_apikey_and_mode()
