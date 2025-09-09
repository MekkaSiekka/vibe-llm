# overview

Provide all the models that are hot-switchable
It downloads the model upon start and save to the local dir, which is mounted locally; each model should have identifiable name and supported languages
The service providing the LLM does not matter, as long as they are open-source

It should have python3 modules that provides the following utils:
- get all available LLMs runnable on this machine
    - thus, check the hardware tflops and compatibility before downloading
- get the each llm availability
- load the llm with the interface provided by the backend server, whatever is provided


