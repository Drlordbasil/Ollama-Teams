import json
import random
import time
import logging
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init
import ollama
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("multi_agent_system")


class AIChatHistory:
    def __init__(self, max_messages: int = 100):
        self.messages: List[Dict[str, str]] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_context(self, last_n: int = 5) -> List[Dict[str, str]]:
        return self.messages[-last_n:]


def generate_response(prompt: str, tools: List[Dict[str, Any]], system_message: str) -> Dict[str, Any]:
    try:
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': prompt}
            ],
            tools=tools
        )
        return response.get('message', {})
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {}


class BaseAgent:
    def __init__(self, name: str, specialty: Optional[str] = None):
        self.name = name
        self.specialty = specialty
        self.memory = AIChatHistory()
        self.logger = logging.getLogger(self.name)

    def think(self) -> Dict[str, Any]:
        context = self.memory.get_context()
        prompt = self.construct_prompt(context)
        tools = self.get_tools()
        system_message = self.get_system_message()
        thought = generate_response(prompt, tools, system_message)
        self.memory.add_message("agent", json.dumps(thought))
        self.logger.info(f"Thought: {thought}")
        return thought

    def act(self, action: Dict[str, Any]) -> Any:
        if 'tool_calls' in action:
            tool_call = action['tool_calls'][0].get('function', {})
            result = self.execute_action(tool_call.get('name', ''), tool_call.get('arguments', {}))
        else:
            result = f"No actionable instruction found in: {action.get('content', '')}"
        self.memory.add_message("environment", f"Action result: {result}")
        self.logger.info(f"Action result: {result}")
        return result

    def execute_action(self, action: str, params: Dict[str, Any]) -> str:
        action_map = self.get_action_map()
        action_func = action_map.get(action)
        if action_func:
            try:
                return action_func(**params)
            except TypeError as te:
                # Attempt to map unexpected arguments to expected ones
                mapped_params = self.map_parameters(action, params)
                if mapped_params:
                    try:
                        return action_func(**mapped_params)
                    except Exception as e:
                        self.logger.error(f"Error executing action '{action}' after mapping: {e}")
                        return f"Error executing action '{action}': {e}"
                self.logger.error(f"Error executing action '{action}': {te}")
                return f"Error executing action '{action}': {te}"
            except Exception as e:
                self.logger.error(f"Unexpected error executing action '{action}': {e}")
                return f"Unexpected error executing action '{action}': {e}"
        return f"Unknown action: {action}"

    def map_parameters(self, action: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Define mappings for known discrepancies
        mappings = {
            'conduct_market_research': {'company_name': 'business_idea'},
            'analyze_data': {'research_topic': 'topic'},
            'review_code': {'existing_code': 'feature_name'},
            'report_findings': {'message': 'findings'}
            # Add more mappings as needed
        }
        if action in mappings:
            expected_params = mappings[action]
            mapped = {expected_params.get(k, k): v for k, v in params.items() if k in expected_params}
            if mapped:
                return mapped
        return None

    def learn(self, experience: Dict[str, Any]) -> None:
        self.memory.add_message("experience", json.dumps(experience))
        self.logger.info("Learned from experience.")

    def run(self, max_iterations: int = 10) -> None:
        for i in range(max_iterations):
            self.logger.info(f"Iteration {i + 1}/{max_iterations}")
            thought = self.think()
            action = self.act(thought)
            experience = {"thought": thought, "action": action}
            self.learn(experience)
            time.sleep(random.uniform(0.5, 2.0))

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def get_tools(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_system_message(self) -> str:
        raise NotImplementedError

    def get_action_map(self) -> Dict[str, Any]:
        raise NotImplementedError


class EntrepreneurAgent(BaseAgent):
    def __init__(self, name: str, business_idea: str):
        super().__init__(name, specialty="Entrepreneurship")
        self.business_idea = business_idea
        self.business_plan: Dict[str, Any] = {}
        self.market_research: Dict[str, Any] = {}
        self.financial_projections: Dict[str, Any] = {}
        self.product_development: Dict[str, Any] = {}
        self.marketing_strategy: Dict[str, Any] = {}

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        return (
            f"As an entrepreneur with the business idea: {self.business_idea}\n"
            f"Given the context:\n{context_str}\n"
            f"What should I focus on next to develop my business?"
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'develop_business_plan',
                    'description': 'Develop or update the business plan',
                    'parameters': {}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'conduct_market_research',
                    'description': 'Conduct or update market research',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'business_idea': {'type': 'string'}
                        },
                        'required': ['business_idea']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'create_financial_projections',
                    'description': 'Create or update financial projections',
                    'parameters': {}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'plan_product_development',
                    'description': 'Plan or update product development strategy',
                    'parameters': {}
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'design_marketing_strategy',
                    'description': 'Design or update marketing strategy',
                    'parameters': {}
                }
            }
        ]

    def get_system_message(self) -> str:
        return "You are an experienced entrepreneur focused on developing innovative business ideas."

    def get_action_map(self) -> Dict[str, Any]:
        return {
            'develop_business_plan': self.develop_business_plan,
            'conduct_market_research': self.conduct_market_research,
            'create_financial_projections': self.create_financial_projections,
            'plan_product_development': self.plan_product_development,
            'design_marketing_strategy': self.design_marketing_strategy
        }

    def develop_business_plan(self) -> str:
        self.business_plan = {
            "executive_summary": "AI-powered personal finance app revolutionizing money management",
            "company_description": "FinGenius: Empowering individuals with AI-driven financial insights",
            "market_analysis": "Growing demand for personalized financial management solutions",
            "organization_management": "Lean team of finance experts and AI engineers",
            "product_line": "AI-powered budgeting, investment advice, and financial forecasting",
            "marketing_sales": "Digital-first approach with focus on mobile app stores and financial forums",
            "funding_request": "Seeking $2M seed funding for development and initial marketing",
            "financial_projections": "Projected break-even in 18 months with 100K active users",
            "appendix": "Detailed technical specifications and market research data"
        }
        return "Business plan developed and updated."

    def conduct_market_research(self, business_idea: str) -> str:
        self.market_research = {
            "target_market": "Millennials and Gen Z individuals seeking financial guidance",
            "market_size": "Estimated $50 billion global personal finance app market by 2028",
            "customer_segments": "Budget-conscious young professionals, gig economy workers, first-time investors",
            "competitors": "Traditional banks' apps, robo-advisors, and established fintech players",
            "market_trends": "Increasing demand for AI-powered financial advice and automation",
            "swot_analysis": (
                "Strengths: AI expertise; Weaknesses: New entrant; "
                "Opportunities: Untapped AI potential in finance; Threats: Regulatory challenges"
            )
        }
        return "Market research conducted and updated."

    def create_financial_projections(self) -> str:
        self.financial_projections = {
            "startup_costs": "$500,000 for initial development and launch",
            "revenue_projections": "Year 1: $1M, Year 2: $3M, Year 3: $8M",
            "expense_projections": "Year 1: $1.5M, Year 2: $2.5M, Year 3: $5M",
            "cash_flow_statement": "Positive cash flow expected by Q4 of Year 2",
            "break_even_analysis": "Break-even point: 100,000 paid subscribers",
            "funding_requirements": "$2M seed funding needed for 18 months runway"
        }
        return "Financial projections created and updated."

    def plan_product_development(self) -> str:
        self.product_development = {
            "product_description": "AI-powered personal finance app with predictive budgeting and investment advice",
            "development_stages": "1. MVP, 2. Beta testing, 3. Public launch, 4. Feature expansion",
            "resources_needed": "AI engineers, mobile developers, UX designers, financial experts",
            "timeline": "6 months to MVP, 3 months beta, public launch in 10 months",
            "testing_strategy": "Closed beta with 1000 users, focus on AI accuracy and user experience",
            "intellectual_property": "Patent pending on AI financial forecasting algorithm"
        }
        return "Product development plan created and updated."

    def design_marketing_strategy(self) -> str:
        self.marketing_strategy = {
            "brand_positioning": "Your AI-powered financial companion for a smarter future",
            "target_audience": "Tech-savvy millennials and Gen Z interested in financial growth",
            "marketing_channels": "Social media, financial blogs, podcast sponsorships, app store optimization",
            "content_strategy": "Educational content on personal finance, AI technology in finance",
            "social_media_strategy": "Influencer partnerships, viral challenges on money-saving tips",
            "budget_allocation": "60% digital ads, 20% content creation, 20% influencer partnerships",
            "kpis": "App downloads, user retention rate, average user savings increase"
        }
        return "Marketing strategy designed and updated."


class DeveloperAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, specialty="Development")
        self.codebase: Dict[str, str] = {}

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        return (
            f"As a developer, given the context:\n{context_str}\n"
            f"What should I focus on next in the development process?"
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'write_code',
                    'description': 'Write or update code for a specific feature',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'feature_name': {'type': 'string'},
                            'code': {'type': 'string'}
                        },
                        'required': ['feature_name', 'code']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'review_code',
                    'description': 'Review and suggest improvements for existing code',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'feature_name': {'type': 'string'}
                        },
                        'required': ['feature_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'integrate_ai_model',
                    'description': 'Integrate an AI model into the application',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'model_name': {'type': 'string'},
                            'integration_code': {'type': 'string'}
                        },
                        'required': ['model_name', 'integration_code']
                    }
                }
            }
        ]

    def get_system_message(self) -> str:
        return "You are an expert developer specializing in AI-powered applications."

    def get_action_map(self) -> Dict[str, Any]:
        return {
            'write_code': self.write_code,
            'review_code': self.review_code,
            'integrate_ai_model': self.integrate_ai_model
        }

    def write_code(self, feature_name: str, code: str) -> str:
        self.codebase[feature_name] = code
        return f"Code written for {feature_name}."

    def review_code(self, feature_name: str) -> str:
        code = self.codebase.get(feature_name, "")
        if not code:
            return f"No code found for feature: {feature_name}."
        review = generate_response(
            f"Review the following code for {feature_name}:\n{code}",
            [],
            ""
        ).get('content', '')
        return f"Code review for {feature_name}: {review}"

    def integrate_ai_model(self, model_name: str, integration_code: str) -> str:
        self.codebase[f"ai_integration_{model_name}"] = integration_code
        return f"AI model {model_name} integrated successfully."


class TesterAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, specialty="Testing")
        self.test_results: Dict[str, str] = {}

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        return (
            f"As a tester, given the context:\n{context_str}\n"
            f"What should I focus on next in the testing process?"
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'run_tests',
                    'description': 'Run tests for a specific feature',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'feature_name': {'type': 'string'}
                        },
                        'required': ['feature_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_test_results',
                    'description': 'Analyze and interpret test results',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'feature_name': {'type': 'string'}
                        },
                        'required': ['feature_name']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'report_bugs',
                    'description': 'Report bugs or issues found during testing',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'feature_name': {'type': 'string'},
                            'bug_description': {'type': 'string'}
                        },
                        'required': ['feature_name', 'bug_description']
                    }
                }
            }
        ]

    def get_system_message(self) -> str:
        return "You are an experienced tester specializing in software quality assurance."

    def get_action_map(self) -> Dict[str, Any]:
        return {
            'run_tests': self.run_tests,
            'analyze_test_results': self.analyze_test_results,
            'report_bugs': self.report_bugs
        }

    def run_tests(self, feature_name: str) -> str:
        code = developer.codebase.get(feature_name, "")
        if not code:
            return f"No code available for feature: {feature_name}."
        test_result = generate_response(
            f"Run tests for the feature '{feature_name}' with the following code:\n{code}",
            [],
            ""
        ).get('content', '')
        self.test_results[feature_name] = test_result
        return f"Tests for {feature_name} completed successfully."

    def analyze_test_results(self, feature_name: str) -> str:
        result = self.test_results.get(feature_name, "")
        if not result:
            return f"No test results found for feature: {feature_name}."
        analysis = generate_response(
            f"Analyze the following test results for {feature_name}:\n{result}",
            [],
            ""
        ).get('content', '')
        return f"Test results for '{feature_name}': {analysis}"

    def report_bugs(self, feature_name: str, bug_description: str) -> str:
        return f"Bug reported for '{feature_name}': {bug_description}."


class ResearchAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, specialty="Research")
        self.research_data: Dict[str, str] = {}

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        return (
            f"As a researcher, given the context:\n{context_str}\n"
            f"What should I focus on next in the research process?"
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'conduct_research',
                    'description': 'Conduct research on a specific topic',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'topic': {'type': 'string'}
                        },
                        'required': ['topic']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_data',
                    'description': 'Analyze and interpret research data',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'topic': {'type': 'string'}
                        },
                        'required': ['topic']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'report_findings',
                    'description': 'Report findings from research',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'topic': {'type': 'string'},
                            'findings': {'type': 'string'}
                        },
                        'required': ['topic', 'findings']
                    }
                }
            }
        ]

    def get_system_message(self) -> str:
        return "You are an experienced researcher specializing in data analysis and interpretation."

    def get_action_map(self) -> Dict[str, Any]:
        return {
            'conduct_research': self.conduct_research,
            'analyze_data': self.analyze_data,
            'report_findings': self.report_findings
        }

    def conduct_research(self, topic: str) -> str:
        research = generate_response(
            f"Conduct comprehensive research on the topic: {topic}.",
            [],
            ""
        ).get('content', '')
        self.research_data[topic] = research
        return f"Research on '{topic}' completed successfully."

    def analyze_data(self, topic: str) -> str:
        data = self.research_data.get(topic, "")
        if not data:
            return f"No research data found for topic: {topic}."
        analysis = generate_response(
            f"Analyze the following research data on {topic}:\n{data}",
            [],
            ""
        ).get('content', '')
        return f"Data analysis for '{topic}': {analysis}"

    def report_findings(self, topic: str, findings: str) -> str:
        return f"Findings reported for '{topic}': {findings}."


class CustomSpecialistAgent(BaseAgent):
    def __init__(self, name: str, specialty: str):
        super().__init__(name, specialty=specialty)
        self.specialist_data: Dict[str, str] = {}

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        return (
            f"As a {self.specialty} specialist, given the context:\n{context_str}\n"
            f"What should I focus on next in my work?"
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'perform_task',
                    'description': f'Perform a task related to {self.specialty}',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task': {'type': 'string'}
                        },
                        'required': ['task']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'analyze_results',
                    'description': f'Analyze and interpret results of {self.specialty} tasks',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task': {'type': 'string'}
                        },
                        'required': ['task']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'report_findings',
                    'description': f'Report findings from {self.specialty} tasks',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'task': {'type': 'string'}
                        },
                        'required': ['task']
                    }
                }
            }
        ]

    def get_system_message(self) -> str:
        return f"You are an experienced {self.specialty} specialist."

    def get_action_map(self) -> Dict[str, Any]:
        return {
            'perform_task': self.perform_task,
            'analyze_results': self.analyze_results,
            'report_findings': self.report_findings
        }

    def perform_task(self, task: str) -> str:
        result = generate_response(
            f"Perform the following {self.specialty} task: {task}.",
            [],
            ""
        ).get('content', '')
        self.specialist_data[task] = result
        return f"{self.specialty} task '{task}' completed successfully."

    def analyze_results(self, task: str) -> str:
        data = self.specialist_data.get(task, "")
        if not data:
            return f"No data found for {self.specialty} task: {task}."
        analysis = generate_response(
            f"Analyze the results of the {self.specialty} task '{task}':\n{data}",
            [],
            ""
        ).get('content', '')
        return f"Results analysis for {self.specialty} task '{task}': {analysis}"

    def report_findings(self, task: str) -> str:
        data = self.specialist_data.get(task, "")
        if not data:
            return f"No data to report for {self.specialty} task: {task}."
        findings = generate_response(
            f"Summarize and report the findings from the {self.specialty} task '{task}':\n{data}",
            [],
            ""
        ).get('content', '')
        return f"Findings reported for {self.specialty} task '{task}': {findings}"


class PeerReviewAgent(BaseAgent):
    def __init__(self, name: str):
        super().__init__(name, specialty="Peer Review")
        self.review_data: Dict[str, str] = {}

    def construct_prompt(self, context: List[Dict[str, str]]) -> str:
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        return (
            f"As a peer reviewer, given the context:\n{context_str}\n"
            f"What should I focus on next in the review process?"
        )

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'review_work',
                    'description': 'Review work from another agent',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'agent_name': {'type': 'string'},
                            'work': {'type': 'string'}
                        },
                        'required': ['agent_name', 'work']
                    }
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'provide_feedback',
                    'description': 'Provide feedback on reviewed work',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'agent_name': {'type': 'string'},
                            'feedback': {'type': 'string'}
                        },
                        'required': ['agent_name', 'feedback']
                    }
                }
            }
        ]

    def get_system_message(self) -> str:
        return "You are an experienced peer reviewer specializing in providing constructive feedback."

    def get_action_map(self) -> Dict[str, Any]:
        return {
            'review_work': self.review_work,
            'provide_feedback': self.provide_feedback
        }

    def review_work(self, agent_name: str, work: str) -> str:
        self.review_data[agent_name] = work
        review = generate_response(
            f"Review the following work from {agent_name}:\n{work}",
            [],
            ""
        ).get('content', '')
        return f"Work from {agent_name} reviewed successfully."

    def provide_feedback(self, agent_name: str, feedback: str) -> str:
        return f"Feedback provided for '{agent_name}': {feedback}."


def display_agent_data(*agents: BaseAgent) -> None:
    for agent in agents:
        print(f"{Fore.RED}[DEBUG] {agent.name} Memory:{Style.RESET_ALL}")
        print(json.dumps(agent.memory.messages, indent=2))
        if isinstance(agent, EntrepreneurAgent):
            print(f"{Fore.RED}[DEBUG] Business Plan:{Style.RESET_ALL}")
            print(json.dumps(agent.business_plan, indent=2))
            print(f"{Fore.RED}[DEBUG] Market Research:{Style.RESET_ALL}")
            print(json.dumps(agent.market_research, indent=2))
            print(f"{Fore.RED}[DEBUG] Financial Projections:{Style.RESET_ALL}")
            print(json.dumps(agent.financial_projections, indent=2))
            print(f"{Fore.RED}[DEBUG] Product Development:{Style.RESET_ALL}")
            print(json.dumps(agent.product_development, indent=2))
            print(f"{Fore.RED}[DEBUG] Marketing Strategy:{Style.RESET_ALL}")
            print(json.dumps(agent.marketing_strategy, indent=2))
        elif isinstance(agent, DeveloperAgent):
            print(f"{Fore.RED}[DEBUG] Developer Agent Codebase:{Style.RESET_ALL}")
            print(json.dumps(agent.codebase, indent=2))
        elif isinstance(agent, TesterAgent):
            print(f"{Fore.RED}[DEBUG] Tester Agent Test Results:{Style.RESET_ALL}")
            print(json.dumps(agent.test_results, indent=2))
        elif isinstance(agent, ResearchAgent):
            print(f"{Fore.RED}[DEBUG] Research Agent Research Data:{Style.RESET_ALL}")
            print(json.dumps(agent.research_data, indent=2))
        elif isinstance(agent, CustomSpecialistAgent):
            print(f"{Fore.RED}[DEBUG] Custom Specialist Agent Data:{Style.RESET_ALL}")
            print(json.dumps(agent.specialist_data, indent=2))
        elif isinstance(agent, PeerReviewAgent):
            print(f"{Fore.RED}[DEBUG] Peer Reviewer Agent Review Data:{Style.RESET_ALL}")
            print(json.dumps(agent.review_data, indent=2))
        print("\n")


def main():
    global entrepreneur, developer, tester, researcher, custom_specialist, peer_reviewer
    entrepreneur = EntrepreneurAgent("EntrepreneurAI", "AI-powered personal finance app")
    developer = DeveloperAgent("DeveloperAI")
    tester = TesterAgent("TesterAI")
    researcher = ResearchAgent("ResearchAI")
    custom_specialist = CustomSpecialistAgent("CustomSpecialistAI", "Financial Analysis")
    peer_reviewer = PeerReviewAgent("PeerReviewerAI")

    agents = [entrepreneur, developer, tester, researcher, custom_specialist, peer_reviewer]

    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {executor.submit(agent.run, max_iterations=10): agent for agent in agents}
        for future in as_completed(futures):
            agent = futures[future]
            try:
                future.result()
                logger.info(f"{agent.name} has completed its run.")
            except Exception as e:
                logger.error(f"{agent.name} generated an exception: {e}")

    display_agent_data(entrepreneur, developer, tester, researcher, custom_specialist, peer_reviewer)


if __name__ == "__main__":
    main()
