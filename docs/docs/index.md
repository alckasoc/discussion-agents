<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

# Welcome to Agential!

## Benchmark Few-shot Examples

| **Benchmarks** | Number of few-shot examples |
| -------------- | --------------------------- |
| **HotpotQA**   |  6                          |
| **FEVER**      |  3                          |
| **TriviaQA**   |  4                          |
| **AmbigNQ**    |  5                          |
| **GSM8k**      |  8                          |
| **SVAMP**      |  7                          |
| **TabMWP**     |  4                          |
| **MBPP**       |  3                          |
| **HumanEval**  |  0                          |
| **ALFWorld**   |                             |
| **WebShop**    |                             |
| **AgentBench** |                             |


## Implementing...



- :octicons-check-16: **:** not tested in the original paper

- :material-check-all: **:** tested in the original paper

| **Methods / Benchmarks** |       HotpotQA       |        FEVER         |       TriviaQA       |       AmbigNQ        |
| ------------------------ | :------------------: | :------------------: | :------------------: | :------------------: |
| ReAct                    | :material-check-all: | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :material-check-all: | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| CRITIC                   | :material-check-all: | :octicons-check-16:  | :material-check-all: | :material-check-all: |
| LATS                     |                      |                      |                      |                      |

| **Methods / Benchmarks** |        GSM8k         |        SVAMP         |        TabMWP        |
| ------------------------ | :------------------: | :------------------: | :------------------: |
| ReAct                    | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :octicons-check-16:  | :octicons-check-16:  | :octicons-check-16:  |
| CRITIC                   | :material-check-all: | :material-check-all: | :material-check-all: |
| LATS                     |                      |                      |                      |

| **Methods / Benchmarks** |         MBPP         |      HumanEval       |
| ------------------------ | :------------------: | :------------------: |
| ReAct                    | :octicons-check-16:  | :octicons-check-16:  |
| Reflexion                | :material-check-all: | :material-check-all: |
| CRITIC                   | :octicons-check-16:  | :octicons-check-16:  |
| LATS                     |                      |                      |

| **Methods / Benchmarks** | ALFWorld | WebShop | AgentBench |
| ------------------------ | :------: | :-----: | :--------: |
| ReAct                    |          |         |            |
| Reflexion                |          |         |            |
| CRITIC                   |          |         |            |
| LATS                     |          |         |            |

## Experimenting...


| **Methods / Benchmarks** | HotpotQA | FEVER | TriviaQA | AmbigNQ |
| ------------------------ | :------: | :---: | :------: | :-----: |
| ReAct                    |          |       |          |         |
| Reflexion                |          |       |          |         |
| CRITIC                   |          |       |          |         |
| LATS                     |          |       |          |         |

| **Methods / Benchmarks** | GSM8k | SVAMP | TabMWP |
| ------------------------ | :---: | :---: | :----: |
| ReAct                    |       |       |        |
| Reflexion                |       |       |        |
| CRITIC                   |       |       |        |
| LATS                     |       |       |        |

| **Methods / Benchmarks** | MBPP  | HumanEval |
| ------------------------ | :---: | :-------: |
| ReAct                    |       |           |
| Reflexion                |       |           |
| CRITIC                   |       |           |
| LATS                     |       |           |

| **Methods / Benchmarks** | ALFWorld | WebShop | AgentBench |
| ------------------------ | :------: | :-----: | :--------: |
| ReAct                    |          |         |            |
| Reflexion                |          |         |            |
| CRITIC                   |          |         |            |
| LATS                     |          |         |            |

## Types of errors

### CRITIC

Sure! Here's the section on CRITIC errors organized into a table for better readability:

## Types of errors

### CRITIC

Certainly! Here's the section organized with each benchmark in a single row and the error types listed in a numbered format within the same cell:

## Types of errors

### CRITIC

| Dataset   | Error Types                                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------------------------ |
| HUMANEVAL | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. No error<br>5. No error                                   |
| MBPP      | 1. Logical error<br>2. Logical error<br>3. Logical error<br>4. No error<br>5. No error                                   |
| GSM8K     | 1. Code efficiency<br>2. NameError (var not defined)<br>3. Logical error<br>4. Logical error<br>5. Logical error         |
| SVAMP     | 1. Logical error<br>2. No error<br>3. Logical error<br>4. No error<br>5. No error                                        |
| TABMWP    | 1. Incorrect answer format<br>2. NameError (var not defined)<br>3. No error<br>4. Incorrect answer format<br>5. No error |
| AMBIGNQ   | 1. No error<br>2. No error<br>3. Incorrect answer/answer format<br>4. No error<br>5. Incorrect answer/answer format      |
| HOTPOTQA  | 1. No error<br>2. No error<br>3. No error<br>4. Incorrect answer/answer format<br>5. Incorrect answer/answer format      |
| TRIVIAQA  | 1. Incorrect answer<br>2. No error<br>3. Incorrect answer<br>4. No error<br>5. No error                                  |
| FEVER     | 1. Incorrect answer<br>2. Incorrect answer format<br>3. No error<br>4. No error<br>5. No error                           |

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
