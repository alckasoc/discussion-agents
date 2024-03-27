
# Discussion Agents


## Features


 Our primary goal is to provide easy-to-use and clean functions for natural language processing tasks, enabling developers to build robust and efficient applications with ease.



- Easy-to-Use Interface: Provides intuitive and user-friendly functions for rapid prototyping and development.

- Clean Functions: Offers clean and well-structured functions, promoting readability and maintainability of code.

- Modularized Implementations: Includes modularized implementations of popular LLM-based agents and agent-related methods, allowing users to leverage cutting-edge innovations from the literature.


## Getting Started 

```python
question = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring?'

llm = ChatOpenAI(openai_api_key="YOUR_API_KEY")
agent = ReActAgent(llm=llm)
out = agent.generate(question=question)
```


Here's what the output would look like!
```
Thought : I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.
Action : Search[best kick boxer in the world]
Observation : (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.
Thought : It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.
Action : Lookup[crimes]
Observation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.
Thought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.
...
```


Project Organization

------------

 

    ├── data
    │   ├── external                   <- Data from third party sources.
    │   ├── interim                    <- Intermediate data that has been transformed.
    │   ├── processed                  <- The final, canonical data sets for modeling.
    │   └── raw                        <- The original, immutable data dump.
    │
    ├── discussion_agents              <- Source code for this project.
    │   ├── cog   
    │   │   ├── agent                  <- Model/agent-related modules.
    │   │   │   
    │   │   ├── eval                   <- Agent core modules.
    │   │   │   
    │   │   ├── functional                  
    │   │   │
    │   │   ├── modules           
    │   │   │   ├── memory             <- Memory-related modules.
    │   │   │   ├── plan               <- Planning-related modules.
    │   │   │   ├── reflect            <- Reflecting-related modules.
    │   │   │   └── score              <- Scoring-related modules.
    │   │   │
    │   │   ├── persona             
    │   │   │
    │   │   └── prompts             
    │   │
    │   └── utils                      <- Utility methods.
    │       
    ├── docs                           <- A default Sphinx project; see sphinx-doc.org for details.
    │
    ├── models                         <- Trained and serialized models, model predictions,
    │                                          or model summaries.
    │       
    ├── notebooks                      <- Jupyter notebooks. Naming convention is a number 
    │                                    (for ordering), the creator's initials, and a short `-` delimited │                                     description, e.g. `1.0-jqp-initial-data-exploration`.
    │  
    │
    ├── references                     <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                        <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                    <- Generated graphics and figures to be used in reporting.
    │
    └── tests                          <- Tests.

---------


## Contribution

If you want to contribute, please check the [contributing.md](https://github.com/alckasoc/discussion-agents/blob/main/CONTRIBUTING.md) for guidelines!
