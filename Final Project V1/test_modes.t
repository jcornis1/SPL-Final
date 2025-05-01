//!------ New feature test: API Key + Mode Support ------!
// This script tests the ability to define and use API keys, modes, and dynamic mode calling.

apikey openai = "sk-test-openai";
apikey huggingface = "hf-test-hfkey";

mode debug uses openai;
mode production uses huggingface;

currentMode = "debug";
callMode("Generate test cases for a string reversal function");

currentMode = "production";
callMode("Explain this code in plain English");