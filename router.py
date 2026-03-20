class Router:
    def __init__(self, llm):
        self.llm = llm
    
    def do_route(self, query, core_content, data_id):
        print(f"data_id: {data_id}, do_route...") 
        
        raw_prompt = open("prompts/route.txt", "r").read()

        prompt = raw_prompt.format(
            query=query,
            titles=core_content
        )
        output = self.llm.response(prompt) 

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

        return chosen