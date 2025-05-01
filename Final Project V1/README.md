
# QuestLang — API Mode-Enabled Programming Language

QuestLang is a custom language built for programmable assistance and dynamic mode switching via API keys.

## 🚀 Features

- 🔐 **API Key Support**: Register and use API keys for any service.
- 🧠 **Modes**: Define modes (e.g., `debug`, `image`, `weather`) and bind them to API keys.
- 🔄 **Dynamic Dispatch**: Switch modes during execution and route prompts accordingly.
- 🧪 **Built-in Testing**: Supports isolated testing via `test_modes.py`.

## 🧬 Language Syntax

### API Key Declaration
```quest
apikey openai = "sk-abc";
apikey weather = "wx-123";
```

### Mode Declaration
```quest
mode debug uses openai;
mode forecast uses weather;
```

### Mode Switching & Call
```quest
currentMode = "debug";
callMode("Improve this function");

currentMode = "forecast";
callMode("Give me the weather in Kent, OH");
```

## 🧪 Testing

Run the mode feature tests:
```bash
python test_modes.py
```

## 📁 Project Structure

- `tokenizer.py` — Lexical analyzer
- `parser.py` — Grammar & AST generation
- `evaluator.py` — Execution engine (with error handling)
- `runner.py` — CLI entry point
- `test_modes.py` — Custom test cases for API/Mode features
- `README.md` — Project documentation

## 💡 Credits

Project idea and enhancements by Brandon Summerlin, inspired by RooCode and intelligent dev assistants like Jarvis.

