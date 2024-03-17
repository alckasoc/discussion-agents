"""ReAct prompts and fewshot examples."""


REACT_INSTRUCTION_HOTPOTQA = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You have a maximum of {max_steps} steps.

Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES = """
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: High Plains refers to one of two distinct land regions:
Thought 4: I need to instead search High Plains (United States).
Action 4: Search[High Plains (United States)]
Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 5: Finish[1,800 to 7,000 ft]
Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]
Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?
Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.
Action 1: Search[Adam Clayton Powell]
Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].
Thought 2: To find the documentary, I can search Adam Clayton Powell (film).
Action 2: Search[Adam Clayton Powell (film)]
Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.
The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.
Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.
Action 3: Finish[The Saimaa Gesture]
Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.
Action 1: Search[Nicholas Ray]
Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.
Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.
Action 2: Search[Elia Kazan]
Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.
Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.
Action 3: Finish[director, screenwriter, actor]
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century. 
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. 
Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. 
Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. 
Action 3: Finish[yes]
"""




REACT_INSTRUCTION_FEVER = """Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFORMATION.  and Action can be two types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""


REACT_WEBTHINK_SIMPLE3_FEVER_EXAMPLES = """
Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Thought 1: I need to search Nikolaj Coster-Waldau and find if he has worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series.. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Thought 2: Because he "appeared in the 2009 Fox television film Virtuality", he should have worked with the Fox Broadcasting Company.
Action 2: Finish[SUPPORTS]

Claim: Stranger Things is set in Bloomington, Indiana.
Thought 1: I should search for Stranger Things, and see if it is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters. 
Thought 2: The observation says that it is set in a "fictional town of Hawkins, Indiana", so it is not set in Bloomington.
Action 2: Finish[REFUTES]

Claim: Beautiful reached number two on the Billboard Hot 100 in 2003.?
Thought 1: I need to search the song Beautiful and find if it reached number two on the Billboard Hot 100 in 2003.
Action 1: Search[Beautiful]
Observation 1: Could not find [Beautiful]. Similar: ['Beautiful', 'Beautiful, Beautiful', 'A Beautiful Mind (film)', 'Beautiful (Christina Aguilera song)', 'Life Is Beautiful'].
Thought 2: From suggestions, I should search "Beautiful (Christina Aguilera song)" to find the song.
Action 2: Search[Beautiful (Christina Aguilera song)]
Observation 2: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002).
Thought 3: It does not mention Billboard, so I need to look up "Billboard Hot 100" to find if it reached number two on it in 2003.
Action 3: Lookup[Billboard Hot 100]
Observation 3: (Result 1 / 3) The song peaked at number two on the Billboard Hot 100 in the United States, where it was certified Gold for 500,000 units shipped.
Thought 4: It only says the song peaked at number two on the Billboard Hot 100, but not if it was in 2003. I am not sure if this claim is true or not.
Action 4: Finish[NOT ENOUGH INFO]
"""


REACT_ALFWORLD_INSTRUCTION = """
{examples}
(END OF EXAMPLES)
{question}
{scratchpad}
"""



REACT_ALFWORLD_PROMPTS_EXAMPLE  =   """

Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, 
a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.,
Thought 1: To solve the task, I need to find and take a sparybottle, then put it on toilet.,
Action 2: OK.,
Thought 2: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.",
Action 3: OK.,
Action 4: go to cabinet 1
Observation 1: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1
Action 5: go to cabinet 2,
Observation 2: The cabinet 2 is closed
Action 6: open cabinet 2,
Observation 3: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.,
Thought 3: Now I find a spraybottle (2). Next, I need to take it.
Action 7: take spraybottle 2 from cabinet 2
Observation 4: You pick up the spraybottle 2 from the cabinet 2.
Thought 4: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.
Action 8: OK.
Action 9: go to toilet 1
Observation 5: On the toilet 1, you see a soapbottle 2.
Action 10: put spraybottle 2 in/on toilet 1
Observation 6: You put the spraybottle 2 in/on the toilet 1.


Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.,
Thought 1: To solve the task, I need to find and take an apple, then put it in sidetable.,
Action 2: OK.,
Thought 2: First I need to find an apple. An apple is more likely to appear in fridges (1), diningtables (1-3), sidetables (1), countertops (1), sinkbasins (1), garbagecan (1). I can check one by one, starting with fridge 1.,
Action 3: OK.,
Action 4: go to fridge 1,
Observation 1: The fridge 1 is closed.,
Action 5: open fridge 1,
Observation 2: You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.,
Action 6: go to diningtable 1,
Observation 3: On the diningtable 1, you see nothing.,
Action 7: go to diningtable 2,
Observation 4: On the diningtable 2, you see a pineapple 1.,
Action 8: go to diningtable 3,
Observation 5: On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.,
Action 9: go to sidetable 1,
Observation 6: On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.,
Action 10: go to countertop 1,
Observation 7: On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.,
Action 11: go to sinkbasin 1,
Observation 8: On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.,
Action 12: go to garbagecan 1,
Observation 9: On the garbagecan 1, you see a apple 3, and a egg 3.,
Thought 3: Now I find an apple (3). Next, I need to take it.,
Action 13: take apple 3 from garbagecan 1,
Observation 10: You pick up the apple 3 from the garbagecan 1.,
Thought 4: Now I find an apple (3). Next, I need to put it in/on sidetable 1.,
Action 14: OK.,
Action 15: go to sidetable 1,
Observation 11: On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.,
Action 16: put apple 3 in/on sidetable 1,
Observation 12: You put the apple 3 in/on the sidetable 1.


Action 1: You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.,
Thought 1: To solve the task, I need to find and take a soapbottle, and put it in garbagecan.,
Action 2: OK.,
Thought 2: First I need to find a soapbottle. A soapbottle is more likely to appear in toilet (1), sinkbasin (1-2), bathtubbasin (1), cabinet (1-4), countertop (1), dresser (1), drawer (1-4), garbagecan (1). I can check one by one, starting with toilet 1.,
Action 3: OK.,
Action 4: go to toilet 1,
Observation 1: On the toilet 1, you see nothing.,
Action 5: go to sinkbasin 1,
Observation 2: On the sinkbasin 1, you see nothing.,
Action 6: go to sinkbasin 2,
Observation 3: On the sinkbasin 2, you see nothing.,
Action 7: go to bathtubbasin 1,
Observation 4: On the bathtubbasin 1, you see a cloth 2.,
Action 8: go to cabinet 1,
Observation 5: The cabinet 1 is closed.,
Action 9: open cabinet 1,
Observation 6: You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.,
Action 10: go to cabinet 2,
Observation 7: The cabinet 2 is closed.,
Action 11: open cabinet 2,
Observation 8: You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.,
Action 12: go to cabinet 3,
Observation 9: The cabinet 3 is closed.,
Action 13: open cabinet 3,
Observation 10: You open the cabinet 3. The cabinet 3 is open. In it, you see a toiletpaper 2.,
Action 14: go to cabinet 4,
Observation 11: The cabinet 4 is closed.,
Action 15: open cabinet 4,
Observation 12: You open the cabinet 4. The cabinet 4 is open. In it, you see a spraybottle 2.,
Action 16: go to countertop 1,
Observation 13: On the countertop 1, you see a soapbottle 2, and a spraybottle 1.,
Thought 3: Now I find a soapbottle (2). Next, I need to take it.,
Action 17: OK.,
Action 18: take soapbottle 2 from countertop 1,
Observation 14: You pick up the soapbottle 2 from the countertop 1.,
Thought 4: Now I take a soapbottle (2). Next, I need to put it in/on garbagecan 1.,
Action 19: OK.,
Action 20: go to garbagecan 1,
Observation 15: On the garbagecan 1, you see nothing.,
Action 21: put soapbottle 2 in/on garbagecan 1,
Observation 16: You put the soapbottle 2 in/on the garbagecan 1.


Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.,
Thought 1: To solve the task, I need to find and take a lettuce, then clean it with sinkbasin, then put it in diningtable.,
Action 2: OK.,
Thought 2: First I need to find a lettuce. A lettuce is more likely to appear in fridge (1), diningtable (1), sinkbasin (1), stoveburner (1-3), cabinet (1-13). I can check one by one, starting with fridge 1.,
Action 3: OK.,
Action 4: go to fridge 1,
Observation 1: The fridge 1 is closed.,
Action 5: open fridge 1,
Observation 2: You open the fridge 1. The fridge 1 is open. In it, you see a cup 3, a egg 2, a potato 3, and a potato 2.,
Action 6: go to diningtable 1,
Observation 3: On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a lettuce 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.,
Thought 3: Now I find a lettuce (1). Next, I need to take it.,
Action 7: OK.,
Action 8: take lettuce 1 from diningtable 1,
Observation 4: You pick up the lettuce 1 from the diningtable 1.,
Thought 4: Now I take a lettuce (1). Next, I need to go to sinkbasin (1) and clean it.,
Action 9: OK.,
Action 10: go to sinkbasin 1,
Observation 5: On the sinkbasin 1, you see a apple 2, a ladle 2, a spoon 1, and a tomato 3.,
Action 11: clean lettuce 1 with sinkbasin 1,
Observation 6: You clean the lettuce 1 using the sinkbasin 1.,
Thought 5: Now I clean a lettuce (1). Next, I need to put it in/on diningtable 1.,
Action 12: OK.,
Action 13: go to diningtable 1,
Observation 7: On the diningtable 1, you see a apple 1, a bread 1, a butterknife 2, a cup 2, a fork 2, a knife 2, a knife 1, a ladle 1, a mug 2, a mug 1, a pan 2, a peppershaker 1, a spatula 3, a tomato 2, and a tomato 1.,
Action 14: put lettuce 1 in/on diningtable 1,
Observation 8: You put the lettuce 1 in/on the diningtable 1.


Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Thought 1: To solve the task, I need to find and take an apple, then clean it with sinkbasin, then put it in sidetable.
Action 2: OK.
Thought 2: First I need to find an apple. An apple is more likely to appear in fridges (1), diningtable (1-3), sidetable (1), countertop (1), sinkbasin (1), garbagecan (1). I can check one by one, starting with fridge 1.
Action 3: OK.
Action 4: go to fridge 1
Observation 1: The fridge 1 is closed.
Action 5: open fridge 1
Observation 2: You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.
Action 6: go to diningtable 1
Observation 3: On the diningtable 1, you see nothing.
Action 7: go to diningtable 2
Observation 4: On the diningtable 2, you see a pineapple 1.
Action 8: go to diningtable 3
Observation 5: On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.
Action 9: go to sidetable 1
Observation 6: On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
Action 10: go to countertop 1
Observation 7: On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.
Action 11: go to sinkbasin 1
Observation 8: On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
Action 12: go to garbagecan 1
Observation 9: On the garbagecan 1, you see a apple 3, and a egg 3.
Thought 3: Now I find an apple (3). Next, I need to take it.
Action 13: OK.
Action 14: take apple 3 from garbagecan 1
Observation 10: You pick up the apple 3 from the garbagecan 1.
Thought 4: Now I take an apple (3). Next, I need to go to sinkbasin (1) and clean it.
Action 15: OK.
Action 16: go to sinkbasin 1
Observation 11: On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
Action 17: clean apple 3 with sinkbasin 1
Observation 12: You clean the apple 3 using the sinkbasin 1.
Thought 5: Now I clean an apple (3). Next, I need to put it in/on sidetable 1.
Action 18: OK.
Action 19: go to sidetable 1
Observation 13: On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
Action 20: put apple 3 in/on sidetable 1
Observation 14: You put the apple 3 in/on the sidetable 1..



Action 1: You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a garbagecan 1, a handtowelholder 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Thought 1: To solve the task, I need to find and take a soapbar, then clean it with sinkbasin, then put it in toilet.
Action 2: OK.
Thought 2: First I need to find a soapbar. A soapbar is more likely to appear in toilet (1), bathtubbasin (1), sinkbasin (1), shelf (1-3), garbagecan (1). I can check one by one, starting with toilet 1.
Action 3: OK.
Action 4: go to toilet 1
Observation 1: On the toilet 1, you see a soapbar 4, and a soapbar 3.
Thought 3: Now I find a soapbar (4). Next, I need to take it.
Action 5: OK.
Action 6: take soapbar 4 from toilet 1
Observation 2: You pick up the soapbar 4 from the toilet 1.
Thought 4: Now I take a soapbar (4). Next, I need to go to sinkbasin (1) and clean it.
Action 7: OK.
Action 8: go to sinkbasin 1
Observation 3: On the sinkbasin 1, you see nothing.
Action 9: clean soapbar 4 with sinkbasin 1
Observation 4: You clean the soapbar 4 using the sinkbasin 1.
Thought 5: Now I clean a soapbar (4). Next, I need to put it in/on toilet 1.
Action 10: OK.
Action 11: go to toilet 1
Observation 5: On the toilet 1, you see a soapbar 3.
Action 12: put soapbar 4 in/on toilet 1
Observation 6: You put the soapbar 4 in/on the toilet 1.


Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Thought 1: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
Action 2: OK.
Thought 2: First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
Action 3: OK.
Action 4: open fridge 1
Observation 1: You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
Action 5: go to countertop 1
Observation 2: On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
Action 6: go to countertop 2
Observation 3: On the countertop 2, you see nothing.
Action 7: go to countertop 3
Observation 4: On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
Thought 3: Now I find an egg (2). Next, I need to take it.
Action 8: OK.
Action 9: take egg 2 from countertop 3
Observation 5: You pick up the egg 2 from the countertop 3.
Thought 4: Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
Action 10: OK.
Action 11: go to microwave 1
Observation 6: The microwave 1 is closed.
Action 12: heat egg 2 with microwave 1
Observation 7: You heat the egg 2 using the microwave 1.
Thought 5: Now I heat an egg (2). Next, I need to put it in/on diningtable 1.
Action 13: OK.
Action 14: go to diningtable 1
Observation 8: On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
Action 15: put egg 2 in/on diningtable 1
Observation 9: You put the egg 2 in/on the diningtable 1.



Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Thought 1: To solve the task, I need to find and take an apple, then heat it with microwave, then put it in fridge.
Action 2: OK.
Thought 2: First I need to find an apple. An apple is more likely to appear in fridge (1), diningtable (1), coffeetable (1), drawer (1), cabinet (1-13), garbagecan (1). I can check one by one, starting with fridge 1.
Action 3: OK.
Action 4: go to fridge 1
Observation 1: The fridge 1 is closed.
Action 5: open fridge 1
Observation 2: You open the fridge 1. The fridge 1 is open. In it, you see a cup 1, and a egg 1.
Action 6: go to diningtable 1
Observation 3: On the diningtable 1, you see a apple 1, a bread 1, a fork 3, a fork 2, a fork 1, a mug 2, a peppershaker 3, a plate 2, a pot 1, a soapbottle 3, a spatula 1, a spoon 3, a spoon 2, a spoon 1, a winebottle 3, and a winebottle 2.
Thought 3: Now I find an apple (1). Next, I need to take it.
Action 7: OK.
Action 8: take apple 1 from diningtable 1
Observation 4: You pick up the apple 1 from the diningtable 1.
Thought 4: Now I take an apple (1). Next, I need to go to a microwave (1) and heat it.
Action 9: OK.
Action 10: go to microwave 1
Observation 5: The microwave 1 is closed.
Action 11: heat apple 1 with microwave 1
Observation 6: You heat the apple 1 using the microwave 1.
Thought 5: Now I heat an apple (1). Next, I need to put it in/on fridge 1.
Action 12: OK.
Action 13: go to fridge 1
Observation 7: The fridge 1 is open. In it, you see a cup 1, and a egg 1.
Action 14: put apple 1 in/on fridge 1
Observation 8: You put the apple 1 in/on the fridge 1.



Action 1: You are in the middle of a room. Looking quickly around you, you see a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Thought 1: To solve the task, I need to find and take a bread, then heat it with microwave, then put it in countertop.
Action 2: OK.
Thought 2: First I need to find a bread. A bread is more likely to appear in fridge (1), toaster (1), diningtable (1), countertop (1-2), drawer (1-3), cabinet (1-10). I can check one by one, starting with fridge 1.
Action 3: OK.
Action 4: go to fridge 1
Observation 1: The fridge 1 is closed.
Action 5: open fridge 1
Observation 2: You open the fridge 1. The fridge 1 is open. In it, you see a mug 3, and a mug 2.
Action 6: go to toaster 1
Observation 3: On the toaster 1, you see nothing.
Action 7: go to diningtable 1
Observation 4: On the diningtable 1, you see a bread 4, a bread 3, a bread 2, a bread 1, a cup 2, a knife 1, a lettuce 1, a potato 3, a tomato 3, a tomato 2, and a winebottle 3.
Thought 3: Now I find a bread (1). Next, I need to take it.
Action 8: OK.
Action 9: take bread 1 from diningtable 1
Observation 5: You pick up the bread 1 from the diningtable 1.
Thought 4: Now I take a bread (1). Next, I need to go to a microwave (1) and heat it.
Action 10: OK.
Action 11: go to microwave 1
Observation 6: The microwave 1 is closed.
Action 12: heat bread 1 with microwave 1
Observation 7: You heat the bread 1 using the microwave 1.
Thought 5: Now I heat a bread (1). Next, I need to put it in/on countertop 1.
Action 13: OK.
Action 14: go to countertop 1
Observation 8: On the countertop 1, you see a apple 1, a egg 1, a pan 2, a peppershaker 3, a plate 2, a plate 1, a soapbottle 1, and a winebottle 1.
Action 15: put bread 1 in/on countertop 1
Observation 9: You put the bread 1 in/on the countertop 1.


Thought: To solve the task, I need to find and take a mug, then cool it with fridge, then put it in shelf.
Action: OK.
Thought: First I need to find a mug. A mug is more likely to appear in countertop (1-3), coffeemachine (1), cabinet (1-9), shelf (1-3), drawer (1-9). I can check one by one, starting with countertop 1.
Action: OK.
Action: Go to countertop 1
Observation: On the countertop 1, you see a knife 1, a pan 2, a pan 1, and a plate 1.
Action: Go to countertop 2
Observation: On the countertop 2, you see a apple 2, a butterknife 2, a fork 3, a houseplant 1, a knife 2, a lettuce 3, a spatula 1, and a winebottle 1.
Action: Go to countertop 3
Observation: On the countertop 3, you see a apple 1, a book 1, a bread 1, a butterknife 3, a glassbottle 2, a glassbottle 1, a knife 3, a lettuce 2, a lettuce 1, a papertowelroll 1, a plate 3, a spoon 1, a statue 1, and a vase 2.
Action: Go to coffeemachine 1
Observation: On the coffeemachine 1, you see nothing.
Action: Go to cabinet 1
Observation: The cabinet 1 is closed.
Action: Open cabinet 1
Observation: You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.
Action: Go to cabinet 2
Observation: The cabinet 2 is closed.
Action: Open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.
Action: Go to cabinet 3
Observation: The cabinet 3 is closed.
Action: Open cabinet 3
Observation: You open the cabinet 3. The cabinet 3 is open. In it, you see a pot 1.
Action: Go to cabinet 4
Observation: On the cabinet 4, you see nothing.
Action: Go to cabinet 5
Observation: The cabinet 5 is closed.
Action: Open cabinet 5
Observation: You open the cabinet 5. The cabinet 5 is open. In it, you see nothing.
Action: Go to cabinet 6
Observation: The cabinet 6 is closed.
Action: Open cabinet 6
Observation: You open the cabinet 6. The cabinet 6 is open. In it, you see a mug 3.
Thought: Now I find a mug (3). Next, I need to take it.
Action: OK.
Action: Take mug 3 from cabinet 6
Observation: You pick up the mug 3 from the cabinet 6.
Thought: Now I take a mug (3). Next, I need to go to a fridge (1) and cool it.
Action: OK.
Action: Go to fridge 1
Observation: The fridge 1 is closed.
Action: Cool mug 3 with fridge 1
Observation: You cool the mug 3 using the fridge 1.
Thought: Now I cool the mug. Next, I need to put it in/on shelf 1.
Action: OK.
Action: Go to shelf 1
Observation: On the shelf 1, you see a cup 2, a cup 1, a peppershaker 1, and a saltshaker 1.
Action: Put mug 3 in/on shelf 1
Observation: You put the mug 3 in/on the shelf 1.

You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1
Thought: To solve the task, I need to find and take a potato, then cool it with fridge, then put it in diningtable.
Action: OK.
Thought: First I need to find a potato. A potato is more likely to appear in fridge (1), diningtable (1), countertop (1-2), cabinet (1-12), drawer (1-3), garbagecan (1). I can check one by one, starting with fridge 1.
Action: OK.
Action: Go to fridge 1
Observation: The fridge 1 is closed.
Action: Open fridge 1
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a apple 1, a bowl 1, a lettuce 2, and a lettuce 1.
Action: Go to diningtable 1
Observation: On the diningtable 1, you see a apple 3, a bread 1, a butterknife 3, a egg 2, a fork 1, a knife 2, a knife 1, a mug 1, a peppershaker 3, a plate 1, a potato 4, a potato 2, a saltshaker 2, a soapbottle 2, and a tomato 3.
Thought: Now I find a potato (2). Next, I need to take it.
Action: OK.
Action: Take potato 2 from diningtable 1
Observation: You pick up the potato 2 from the diningtable 1.
Thought: Now I take a potato (2). Next, I need to go to a fridge (1) and cool it.
Action: OK.
Action: Go to fridge 1
Observation: The fridge 1 is open. In it, you see a apple 1, a bowl 1, a lettuce 2, and a lettuce 1.
Action: Cool potato 2 with fridge 1
Observation: You cool the potato 2 using the fridge 1.
Thought: Now I cool a potato (2). Next, I need to put it in/on diningtable 1.
Action: OK.
Action: Go to diningtable 1
Observation: On the diningtable 1, you see a apple 3, a bread 1, a butterknife 3, a egg 2, a fork 1, a knife 2, a knife 1, a mug 1, a peppershaker 3, a plate 1, a potato 4, a saltshaker 2, a soapbottle 2, and a tomato 3.
Action: Put potato 2 in/on diningtable 1
Observation: You put the potato 2 in/on the diningtable 1.

You are in the middle of a room. Looking quickly around you, you see a cabinet 20, a cabinet 19, a cabinet 18, a cabinet 17, a cabinet 16, a cabinet 15, a cabinet 14, a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Thought: To solve the task, I need to find and take the first creditcard, then put it in dresser, then find and take the second creditcard, then put it in dresser.
Action: OK.
Thought: First I need to find the first creditcard. A creditcard is more likely to appear in drawer (1-2), coutertop (1), sidetable (1), diningtable (1), armchair (1-2), bed (1). I can check one by one, starting with drawer 1.
Action: OK.
Action: Go to drawer 1
Observation: The drawer 1 is closed.
Action: Open drawer 1
Observation: You open the drawer 1. The drawer 1 is open. In it, you see a book 1, a cd 1, and a pen 1.
Action: Go to drawer 2
Observation: The drawer 2 is closed.
Action: Open drawer 2
Observation: You open the drawer 2. The drawer 2 is open. In it, you see nothing.
Action: Go to countertop 1
Observation: On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a creditcard 2, a mirror 1, a pencil 2, and a pencil 1.
Thought: Now I find the first creditcard (2). Next, I need to take it. I can find the second creditcard (3) later in countertop 1.
Action: OK.
Action: Take creditcard 2 from countertop 1
Observation: You pick up the creditcard 2 from the countertop 1.
Thought: Now I take the first creditcard (2). Next, I need to put it in/on dresser 1.
Action: OK.
Action: Go to dresser 1
Observation: On the dresser 1, you see a mug 1, and a television 1.
Action: Put creditcard 2 in/on dresser 1
Observation: You put the creditcard 2 in/on the dresser 1.
Thought: Now I put the first creditcard in dresser. Next, I need to find the second creditcard. I can directly go to countertop 1.
Action: OK.
Action: Go to countertop 1
Observation: On the countertop 1, you see a cellphone 2, a creditcard 4, a creditcard 3, a mirror 1, a pencil 2, and a pencil 1.
Thought: Now I find the second creditcard (3). Next, I need to take it.
Action: OK.
Action: Take creditcard 3 from countertop 1
Observation: You pick up the creditcard 3 from the countertop 1.
Thought: Now I take the second creditcard (3). Next, I need to put it in/on dresser 1.
Action: OK.
Action: Go to dresser 1
Observation: On the dresser 1, you see a creditcard 2, a mug 1, and a television 1.
Action: Put creditcard 3 in/on dresser 1
Observation: You put the creditcard 3 in/on the dresser 1.

You are in the middle of a room. Looking quickly around you, you see a bed 1, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a shelf 9, a shelf 8, a shelf 7, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Thought: To solve the task, I need to find and take a pen, then find and use a desklamp. First I need to find a pen. A pen is more likely to appear in drawer (1-10), shelf (1-9), bed (1), garbagecan (1). I can check one by one, starting with drawer 1.
Action: OK.
Action: Go to drawer 1
Observation: The drawer 1 is closed.
Action: Open drawer 1
Observation: You open the drawer 1. The drawer 1 is open. In it, you see nothing.
Action: Go to drawer 4
Observation: The drawer 4 is closed.
Action: Open drawer 4
Observation: You open the drawer 4. The drawer 4 is open. In it, you see a cd 3, a keychain 1, and a pen 2.
Thought: Now I find a pen (2). Next, I need to take it.
Action: OK.
Action: Take pen 2 from drawer 4
Observation: You pick up the pen 2 from the drawer 4.
Thought: Now I take a pen (2). Next, I need to find a desklamp. A desklamp is more likely to appear in dresser (1), shelf (1-9), bed (1), garbagecan (1), drawer (1-10). I can check one by one, starting with dresser 1.
Action: OK.
Action: Go to dresser 1
Observation: On the dresser 1, you see a alarmclock 2, a alarmclock 1, a book 1, a cd 1, a creditcard 1, a desklamp 1, a keychain 2, a pen 3, and a pen 1.
Thought: Now I find a desklamp (1). Next, I need to use it.
Action: OK.
Action: Use desklamp 1
Observation: You turn on the desklamp 1.

You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1.
Thought: To solve the task, I need to find and take a statue, then find and use a desklamp.
Action: OK.
Thought: First I need to find a statue. A statue is more likely to appear in dresser (1), coffeetable (1), sidetable (1-2), diningtable (1), sofa (1), garbagecan (1). I can check one by one, starting with dresser 1.
Action: OK.
Action: Go to dresser 1
Observation: On the dresser 1, you see a cellphone 3, a newspaper 2, a statue 1, and a television 1.
Thought: Now I find a statue (1). Next, I need to take it.
Action: OK.
Action: Take statue 1 from dresser 1
Observation: You pick up the statue 1 from the dresser 1.
Thought: Now I take a statue (1). Next, I need to find a desklamp. A desklamp is more likely to appear in sidetable (1-2), coffeetable (1), diningtable (1). I can check one by one, starting with sidetable 1.
Action: OK.
Action: Go to sidetable 1
Observation: On the sidetable 1, you see nothing.
Action: Go to sidetable 2
Observation: On the sidetable 2, you see a desklamp 3, a newspaper 1, and a statue 2.
Thought: Now I find a desklamp (3). Next, I need to use it.
Action: OK.
Action: Use desklamp 3
Observation: You turn on the desklamp 3.

You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Action: Go to cabinet 1
Observation: On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.
Action: Go to cabinet 2
Observation: The cabinet 2 is closed.
Action: Open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.
Action: Take spraybottle 2 from cabinet 2
Observation: You pick up the spraybottle 2 from the cabinet 2.
Action: Go to toilet 1
Observation: On the toilet 1, you see a soapbottle 2.
Action: Put spraybottle 2 in/on toilet 1
Observation: You put the spraybottle 2 in/on the toilet 1.

You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Action: Go to fridge 1
Observation: The fridge 1 is closed.
Action: Open fridge 1
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.
Action: Go to diningtable 1
Observation: On the diningtable 1, you see nothing.
Action: Go to diningtable 2
Observation: On the diningtable 2, you see a pineapple 1.
Action: Go to diningtable 3
Observation: On the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.
Action: Go to sidetable 1
Observation: On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
Action: Go to countertop 1
Observation: On the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.
Action: Go to sinkbasin 1
Observation: On the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.
Action: Go to garbagecan 1
Observation: On the garbagecan 1, you see a apple 3, and a egg 3.
Action: Take apple 3 from garbagecan 1
Observation: You pick up the apple 3 from the garbagecan 1.
Action: Go to sidetable 1
Observation: On the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.
Action: Put apple 3 in/on sidetable 1
Observation: You put the apple 3 in/on the sidetable 1.

You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.
Action: Go to toilet 1
Observation: On the toilet 1, you see nothing.
Action: Go to sinkbasin 1
Observation: On the sinkbasin 1, you see nothing.
Action: Go to sinkbasin 2
Observation: On the sinkbasin 2, you see nothing.
Action: Go to bathtubbasin 1
Observation: On the bathtubbasin 1, you see a cloth 2.
Action: Go to cabinet 1
Observation: The cabinet 1 is closed.
Action: Open cabinet 1
Observation: You open the cabinet 1. The cabinet 1 is open. In it, you see nothing.
Action: Go to cabinet 2
Observation: The cabinet 2 is closed.
Action: Open cabinet 2
Observation: You open the cabinet 2. The cabinet 2 is open. In it, you see nothing.
Action: Go to cabinet 3
Observation: The cabinet 3 is closed.
Action: Open cabinet 3
Observation: You open the cabinet 3. The cabinet 3 is open. In it, you see a toiletpaper 2.
Action: Go to cabinet 4
Observation: The cabinet 4 is closed.
Action: Open cabinet 4
Observation: You open the cabinet 4. The cabinet 4 is open. In it, you see a spraybottle 2.
Action: Go to countertop 1
Observation: On the countertop 1, you see a soapbottle 2, and a spraybottle 1.
Action: Take soapbottle 2 from countertop 1
Observation: You pick up the soapbottle 2 from the countertop 1.
Action: Go to garbagecan 1
Observation: On the garbagecan 1, you see nothing.
Action: Put soapbottle 2 in/on garbagecan 1
Observation: You put the soapbottle 2 in/on the garbagecan 1.

Action: You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put a clean lettuce in diningtable.

Thought: To accomplish this task, I need to locate a lettuce, clean it using the sinkbasin, and then place it on the diningtable.

Observation: The diningtable already has a lettuce 1 among other items like apple 1, bread 1, butterknife 2, etc.

Action: > go to fridge 1
Thought: I don't need to go to the fridge since the lettuce is already present on the diningtable.
Observation: The fridge is closed.

Action: > open fridge 1
Thought: Again, there's no need to open the fridge as the lettuce is on the diningtable.
Observation: The fridge contains a cup 3, an egg 2, a potato 3, and a potato 2.

Action: > go to diningtable 1
Thought: Going to the diningtable is the right step since the lettuce is there.
Observation: The diningtable has various items including the required lettuce 1.

Action: > take lettuce 1 from diningtable 1
Thought: Taking the lettuce from the diningtable is the correct next action.
Observation: I have picked up the lettuce 1 from the diningtable.

Action: > go to sinkbasin 1
Thought: Going to the sinkbasin is necessary to clean the lettuce.
Observation: The sinkbasin has an apple 2, a ladle 2, a spoon 1, and a tomato 3 on it.

Action: > clean lettuce 1 with sinkbasin 1
Thought: Cleaning the lettuce using the sinkbasin is the required step.
Observation: I have cleaned the lettuce 1 using the sinkbasin.

Action: > go to diningtable 1
Thought: Now that the lettuce is clean, I need to go back to the diningtable.
Observation: The diningtable still has the same items as before, except the cleaned lettuce.

Action: > put lettuce 1 in/on diningtable 1
Thought: Placing the clean lettuce on the diningtable completes the task.
Observation: I have put the clean lettuce 1 on the diningtable.


Action: You are in the middle of a room. Looking quickly around you, you see a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put a hot apple in fridge.

Thought: To complete this task, I need to find an apple, heat it up using the microwave, and then put the hot apple in the fridge.

Observation: The fridge contains a cup 1 and an egg 1. The diningtable has various items including an apple 1.

Action: > go to fridge 1
Thought: I don't need to go to the fridge yet since I haven't heated the apple.
Observation: The fridge is closed.

Action: > open fridge 1
Thought: Again, no need to open the fridge until after heating the apple.
Observation: The fridge now shows its contents after being opened.

Action: > go to diningtable 1
Thought: Going to the diningtable is the right step to get the apple.
Observation: The diningtable has an apple 1 among other items.

Action: > take apple 1 from diningtable 1
Thought: Taking the apple from the diningtable is the correct next action.
Observation: I have picked up the apple 1 from the diningtable.

Action: > go to microwave 1
Thought: Going to the microwave is necessary to heat the apple.
Observation: The microwave is closed.

Action: > heat apple 1 with microwave 1
Thought: Heating the apple using the microwave is the required step.
Observation: I have heated the apple 1 using the microwave.

Action: > go to fridge 1
Thought: Now that the apple is hot, I need to go to the fridge.
Observation: The fridge is open and still contains the cup 1 and egg 1.

Action: > put apple 1 in/on fridge 1
Thought: Putting the hot apple in the fridge completes the task.
Observation: I have put the hot apple 1 in the fridge.


Action: You are in the middle of a room. Looking quickly around you, you see a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some bread and put it in countertop.

Thought: To accomplish this, I need to locate some bread, heat it up using the microwave, and then place the hot bread on one of the countertops.

Observation: The fridge contains a mug 3 and a mug 2. The diningtable has breads and other items.

Action: > go to fridge 1
Thought: I don't need to go to the fridge since there's no bread in there.
Observation: The fridge is closed.

Action: > open fridge 1
Thought: Again, no need to open the fridge as it doesn't contain any bread.
Observation: The fridge shows its contents after being opened.

Action: > go to toaster 1
Thought: The toaster is likely not useful for heating bread, so I'll skip it.
Observation: The toaster is empty.

Action: > go to diningtable 1
Thought: Going to the diningtable is the right step to get the bread.
Observation: The diningtable has breads (bread 4, bread 3, bread 2, bread 1) among other items.

Action: > take bread 1 from diningtable 1
Thought: Taking one of the breads from the diningtable is the correct next action.
Observation: I have picked up the bread 1 from the diningtable.

Action: > go to microwave 1
Thought: Going to the microwave is necessary to heat the bread.
Observation: The microwave is closed.

Action: > heat bread 1 with microwave 1
Thought: Heating the bread using the microwave is the required step.
Observation: I have heated the bread 1 using the microwave.

Action: > go to countertop 1
Thought: Now that the bread is hot, I need to go to one of the countertops.
Observation: The countertop has various items like an apple 1, egg 1, pans, plates, etc.

Action: > put bread 1 in/on countertop 1
Thought: Putting the hot bread on the countertop completes the task.
Observation: I have put the hot bread 1 on the countertop.

Action: You are in the middle of a room. Looking quickly around you, you see a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: put a cool mug in shelf.

Thought: To accomplish this task, I need to find a mug, cool it down using the fridge, and then place the cool mug on one of the shelves.

Observation: The countertops have various items like knives, pans, plates, fruits, and utensils. The cabinets seem to be the likely place to find a mug.

Action: > go to countertop 1
Thought: Checking the countertops first to see if there's a mug, but it doesn't seem likely.
Observation: The countertop has a knife 1, pans, and a plate 1, but no mug.

Action: > go to countertop 2
Thought: Checking the other countertops as well, but still no sign of a mug.
Observation: This countertop has fruits, utensils, but no mug.

Action: > go to countertop 3
Thought: Checking the last countertop, but it's unlikely to have a mug.
Observation: This countertop has various items like fruits, bread, utensils, but no mug.

Action: > go to coffeemachine 1
Thought: The coffeemachine might have a mug nearby, so it's worth checking.
Observation: The coffeemachine is empty.

Action: > go to cabinet 1
Thought: Checking the cabinets now, as they seem like a logical place to find a mug.
Observation: The cabinet is closed.

... (skipping some steps for brevity) ...

Action: > open cabinet 6
Thought: Opening each cabinet until I find one with a mug.
Observation: The cabinet contains a mug 3.

Action: > take mug 3 from cabinet 6
Thought: Taking the mug from the cabinet is the correct next action.
Observation: I have picked up the mug 3 from the cabinet.

Action: > go to fridge 1
Thought: Going to the fridge to cool down the mug.
Observation: The fridge is closed.

Action: > cool mug 3 with fridge 1
Thought: Cooling the mug using the fridge is the required step.
Observation: I have cooled the mug 3 using the fridge.

Action: > go to shelf 1
Thought: Now that the mug is cool, I need to go to one of the shelves.
Observation: The shelf has cups, a peppershaker, and a saltshaker on it.

Action: > put mug 3 in/on shelf 1
Thought: Placing the cool mug on the shelf completes the task.
Observation: I have put the cool mug 3 on the shelf.


Action: You are in the middle of a room. Looking quickly around you, you see a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: cool some potato and put it in diningtable.

Thought: To complete this task, I need to locate a potato, cool it down using the fridge, and then place the cool potato on the diningtable.

Observation: The fridge contains some apples, a bowl, and lettuces. The diningtable has various items including potatoes (potato 4, potato 2).

Action: > go to fridge 1
Thought: I don't need to go to the fridge yet since I haven't identified the potato to cool.
Observation: The fridge is closed.

Action: > open fridge 1
Thought: Again, no need to open the fridge until after getting the potato.
Observation: The fridge shows its contents after being opened.

Action: > go to diningtable 1
Thought: Going to the diningtable is the right step to get the potato.
Observation: The diningtable has potatoes among other items.

Action: > take potato 2 from diningtable 1
Thought: Taking one of the potatoes from the diningtable is the correct next action.
Observation: I have picked up the potato 2 from the diningtable.

Action: > go to fridge 1
Thought: Now that I have the potato, I need to go to the fridge to cool it down.
Observation: The fridge is open and still contains its previous contents.

Action: > cool potato 2 with fridge 1
Thought: Cooling the potato using the fridge is the required step.
Observation: I have cooled the potato 2 using the fridge.

Action: > go to diningtable 1
Thought: Now that the potato is cool, I need to go back to the diningtable.
Observation: The diningtable still has the same items as before, except the cool potato.

Action: > put potato 2 in/on diningtable 1
Thought: Placing the cool potato on the diningtable completes the task.
Observation: I have put the cool potato 2 on the diningtable

Action: You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1.
Your task is to: put two cellphone in sofa.

Thought: To accomplish this task, I need to locate two cellphones and place them on the sofa.

Observation: The coffeetable has a cellphone 3 among other items.

Action: > go to coffeetable 1
Thought: Going to the coffeetable is a good first step to see if there are any cellphones there.
Observation: The coffeetable has a box 1, a cellphone 3, and a plate 1.

Action: > take cellphone 3 from coffeetable 1
Thought: Taking the cellphone 3 from the coffeetable is the correct next action.
Observation: I have picked up the cellphone 3 from the coffeetable.

Action: > go to sofa 1
Thought: Now that I have one cellphone, I need to go to the sofa to put it there.
Observation: The sofa has a newspaper 2, a pillow 1, remotecontrols, but no cellphones yet.

Action: > put cellphone 3 in/on sofa 1
Thought: Putting the first cellphone on the sofa is the right step.
Observation: I have put the cellphone 3 on the sofa.

Action: > go to diningtable 1
Thought: To find the second cellphone, I should check the diningtable.
Observation: The diningtable has a cellphone 2 among other items.

Action: > take cellphone 2 from diningtable 1
Thought: Taking the cellphone 2 from the diningtable is the correct next action.
Observation: I have picked up the cellphone 2 from the diningtable.

Action: > go to sofa 1
Thought: Now that I have both cellphones, I need to go back to the sofa.
Observation: The sofa now has the cellphone 3 and other items.

Action: > put cellphone 2 in/on sofa 1
Thought: Putting the second cellphone on the sofa completes the task.
Observation: I have put the cellphone 2 on the sofa, along with the cellphone 3.


You are in the middle of a room. Looking quickly around you, you see a coffeetable 1, a diningtable 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a dresser 1, a garbagecan 1, a sidetable 2, a sidetable 1, and a sofa 1.

Action: Your task is to put two cellphones on the sofa.
Thought: Okay, I need to locate two cellphones and then put them on the sofa.
Observation: On the coffeetable 1, you see a box 1, a cellphone 3, and a plate 1.

Action: Go to coffeetable 1
Action: Take cellphone 3 from coffeetable 1

Thought: Got one cellphone, need to find another.
Observation: On the diningtable 1, you see a cellphone 2, a keychain 2, a laptop 1, a statue 2, and a statue 1.

Action: Go to diningtable 1
Action: Take cellphone 2 from diningtable 1
Thought: Great, now I have two cellphones.

Observation: On the sofa 1, you see a newspaper 2, a pillow 1, a remotecontrol 2, and a remotecontrol 1.

Action: Go to sofa 1
Action: Put cellphone 3 on sofa 1
Action: Put cellphone 2 on sofa 1
Thought: Both cellphones are now successfully put on the sofa 1.
Observation: The sofa 1 now has the cellphone 3, cellphone 2, newspaper 2, pillow 1, remotecontrol 2, and remotecontrol 1 on it.
Thought: The task is complete!


You are in the middle of a room. Looking quickly around you, you see a lot of cabinets, countertops, drawers, a fridge, shelves, etc. Seems like a kitchen area.

Action: Your task is to put two saltshakers in a drawer.
Thought: Okay, I need to locate two saltshakers and an empty drawer to put them in.
Observation: The drawers 1, 2, 3 are currently closed. Various items on the countertops.

Action: Go to drawer 1
Observation: The drawer 1 is closed.
Action: Open drawer 1
Observation: Drawer 1 is now open and empty inside.

Action: Go to drawer 2
Observation: Drawer 2 is closed.
Action: Open drawer 2
Observation: Drawer 2 is open with a dishsponge 3 inside.

Action: Go to drawer 3
Observation: Drawer 3 is closed.

Action: Open drawer 3
Observation: Drawer 3 is open with a fork 2, ladle 2, and spoon 1 inside.

Action: Go to countertop 3
Observation: On countertop 3, you see a saltshaker 4 among other items.
Action: Take saltshaker 4 from countertop 3
Thought: Got one saltshaker, need one more.

Action: Go to drawer 1
Observation: Drawer 1 is open and has nothing in it currently.
Action: Put saltshaker 4 in drawer 1
Thought: Okay, one saltshaker in the drawer.

Action: Go to cabinet 3
Observation: Cabinet 3 is closed.
Action: Open cabinet 3

Observation: Cabinet 3 is open with a saltshaker 3 and saltshaker 2 inside.
Action: Take saltshaker 2 from cabinet 3

Action: Go to drawer 1
Observation: Drawer 1 is open with the saltshaker 4 inside.
Action: Put saltshaker 2 in drawer 1
Thought: Both saltshakers are now successfully placed in drawer 1!
Observation: Drawer 1 contains the saltshaker 4 and saltshaker 2.
Thought: The task is complete.


You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 1, drawers 1-3, a garbagecan 1, shelves 1-5, and some sidetables.

Action: Your task is to look at a bowl under the desk lamp.
Thought: Okay, I need to find a bowl and get it positioned under a desk lamp to examine it closely.
Observation: On the desk 1, you see an alarmclock 2, book 1, cellphone 1, keychain 1, laptops 1 & 2, and pen 3.

Action: Go to drawer 1
Observation: Drawer 1 is empty.

Action: Go to drawer 2
Observation: Drawer 2 is closed.
Action: Open drawer 2
Observation: Drawer 2 is open but empty inside.

Action: Go to drawer 3
Observation: Drawer 3 is closed.
Action: Open drawer 3
Observation: Drawer 3 is open but empty inside.

Action: Go to shelf 2

Observation: On shelf 2, you see a bowl 1.
Action: Take bowl 1 from shelf 2
Thought: Got the bowl, now need to find the desk lamp.

Action: Go to sidetable 2
Observation: On sidetable 2, you see an alarmclock 1, a desklamp 1, and a pen 2.

Action: Use desklamp 1
Thought: Okay, the desklamp is now on.
Observation: You examine the bowl 1 under the lit desklamp 1.

You are in the middle of a room. Looking quickly around you, you see a bed 1, drawers 1-10, a dresser 1, a garbagecan 1, and shelves 1-9.

Action: Your task is to examine a pen with the desk lamp.
Thought: Okay, I need to find a pen and a desk lamp, then use the lamp to closely examine the pen.
Observation: The drawers 1-3 are closed.

Action: Go to drawer 1
Observation: Drawer 1 is closed.
Action: Open drawer 1
Observation: Drawer 1 is open but empty inside.

Action: Go to drawer 2

Observation: Drawer 2 is closed.
Action: Open drawer 2
Observation: Drawer 2 is open but empty inside.

Action: Go to drawer 3
Observation: Drawer 3 is closed.
Action: Open drawer 3
Observation: Drawer 3 is open but empty inside.

Action: Go to drawer 4
Observation: Drawer 4 is closed.
Action: Open drawer 4
Observation: Drawer 4 is open with a cd 3, keychain 1, and pen 2 inside.
Action: Take pen 2 from drawer 4
Thought: Got the pen, now need to find the desk lamp.

Action: Go to dresser 1
Observation: On dresser 1 you see alarm clocks, books, cds, credit card, desk lamp 1, keychains and pens.
Action: Use desklamp 1
Thought: Okay, the desk lamp is on now.
Observation: You examine the pen 2 closely under the lit desklamp 1.
Thought: The task to examine the pen with the desk lamp is now complete!


You are in the middle of a room. Looking quickly around you, you see a coffee table, dining table, drawers, dresser, side tables, sofa, etc.

Action: Your task is to look at a statue under the desk lamp.
Thought: I need to locate a statue and a desk lamp, then position the statue under the lamp to examine it.
Observation: On the dresser 1, you see a cellphone 3, newspaper 2, statue 1, and television 1.

Action: Go to dresser 1
Action: Take statue 1 from dresser 1

Thought: Got the statue, now need to find the desk lamp.
Observation: On sidetable 1, there is nothing. On sidetable 2, you see a desklamp 3, newspaper 1, and statue 2.

Action: Go to sidetable 2
Action: Use desklamp 3
Thought: The desk lamp is now on.
Observation: You position the statue 1 under the lit desklamp 3 and examine it closely.
Thought: The task to look at the statue under the desk lamp is complete!

"""
