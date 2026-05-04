class Router:
    def __init__(self, llm):
        self.llm = llm
        self.max_new_tokens = int(__import__("os").environ.get("STRUCTRAG_ROUTER_MAX_NEW_TOKENS", "128"))

    def _prompt_path(self):
        prompt_style = __import__("os").environ.get("STRUCTRAG_ROUTER_PROMPT_STYLE", "fewshot")
        if prompt_style == "learned":
            return "prompts/route_learned.txt"
        return "prompts/route.txt"
    
    def do_route(self, query, core_content, data_id):
        print(f"data_id: {data_id}, do_route...") 
        
        prompt_path = self._prompt_path()
        raw_prompt = open(prompt_path, "r").read()

        prompt = raw_prompt.format(
            query=query,
            titles=core_content
        )
        output = self.llm.response(
            prompt,
            max_new_tokens=self.max_new_tokens,
            trace_context={
                "data_id": data_id,
                "component": "router.do_route",
                "metadata": {
                    "prompt_path": prompt_path,
                    "max_new_tokens": self.max_new_tokens,
                },
            },
        ) 

        if "table" in output.lower():
            chosen = "table"
        elif "graph" in output.lower():
            chosen = "graph"
        elif "algorithm" in output.lower():
            chosen = "algorithm"
        elif "catalogue" in output.lower():
            chosen = "catalogue"
        else:
            chosen = "chunk"

        return chosen, output
