"""
Enhanced Calculator Tool with Retry Strategies
-------------------------------------------
Purpose: Provides a robust calculator implementation for Claude AI with multiple retry
strategies for handling failures and word problems.

Requirements:
- anthropic
- python-dotenv

Environment Variables:
- ANTHROPIC_API_KEY must be set in .env file

Usage:
1. Set up environment variables
2. Run script directly for tests
3. Import and use call_claude_with_calculator() function for calculations

Bash command to run:
python3 calculator.py
"""

import os
import re
import time
import logging
import threading
from typing import Dict, Union, List, Tuple, Any, Optional
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Tool Definition
calculator_tool = {
    "name": "calculator",
    "description": "A simple calculator that performs basic arithmetic operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform."
            },
            "operand1": {
                "type": "number",
                "description": "The first operand."
            },
            "operand2": {
                "type": "number",
                "description": "The second operand."
            }
        },
        "required": ["operation", "operand1", "operand2"]
    }
}

def calculator(operation: str, operand1: float, operand2: float) -> float:
    """Basic calculator function to perform arithmetic operations"""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    
    if operation not in operations:
        raise ValueError(f"Unsupported operation: {operation}")
        
    return operations[operation](operand1, operand2)

def execute_calculator_tool(params: dict) -> dict:
    """Execute calculator with provided parameters"""
    try:
        result = calculator(
            operation=params["operation"],
            operand1=params["operand1"],
            operand2=params["operand2"]
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ValidationRetryStrategy:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.validation_ranges = {
            "add": (-1e10, 1e10),
            "subtract": (-1e10, 1e10),
            "multiply": (-1e15, 1e15),
            "divide": (-1e10, 1e10)
        }

    def extract_numbers_from_text(self, text: str) -> List[float]:
        """Extract numbers from text for word problems"""
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return [float(num) for num in numbers]
    
    def determine_operation_from_text(self, text: str) -> Optional[str]:
        """Determine the arithmetic operation from text"""
        text = text.lower()
        operations = {
            "add": ["add", "plus", "sum", "increase", "more", "gain", "additional"],
            "subtract": ["subtract", "minus", "difference", "decrease", "less", "lose", "give away", "remain", "left"],
            "multiply": ["multiply", "times", "product", "multiplication"],
            "divide": ["divide", "split", "ratio", "per", "division"]
        }
        
        for op, keywords in operations.items():
            if any(keyword in text for keyword in keywords):
                return op
        return None

    def validate_result(self, result: Dict[str, Any], operation: str) -> bool:
        """Validate calculation results"""
        if "error" in result or "result" not in result:
            return False
            
        value = result["result"]
        if not isinstance(value, (int, float)):
            return False
            
        min_val, max_val = self.validation_ranges.get(operation, (-float('inf'), float('inf')))
        return min_val <= value <= max_val

    def call_with_retry(self, client: Anthropic, user_input: str, calculator_tool: dict) -> Dict[str, Any]:
        attempts = 0
        last_error = None
        
        logger.info(f"\n{'='*50}\nStarting validation retry strategy for input: {user_input}")
        
        while attempts < self.max_retries:
            try:
                logger.info(f"\n----- Attempt {attempts + 1}/{self.max_retries} -----")
                system_message = (
                    "You are a calculation assistant. For all mathematical operations, "
                    "use the calculator tool. For word problems, first identify the "
                    "numbers and operation, then use the calculator tool to solve it. "
                    "Never perform calculations yourself."
                )
                if last_error:
                    system_message += f" Previous attempt failed: {last_error}"
                
                logger.info(f"System message: {system_message}")
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    temperature=0,
                    system=system_message,
                    messages=[{"role": "user", "content": user_input}],
                    tools=[calculator_tool]
                )
                
                logger.info(f"Claude response: {response.content}")
                logger.info(f"Stop reason: {response.stop_reason}")
                
                if response.stop_reason != "tool_use":
                    numbers = self.extract_numbers_from_text(user_input)
                    operation = self.determine_operation_from_text(user_input)
                    
                    logger.info(f"Extracted numbers: {numbers}")
                    logger.info(f"Determined operation: {operation}")
                    
                    if numbers and operation and len(numbers) >= 2:
                        result = execute_calculator_tool({
                            "operation": operation,
                            "operand1": numbers[0],
                            "operand2": numbers[1]
                        })
                        logger.info(f"Calculator result: {result}")
                        if self.validate_result(result, operation):
                            return result
                    last_error = "Could not extract calculation from text"
                    logger.warning(f"Attempt {attempts + 1} failed: {last_error}")
                else:
                    tool_use = next((block for block in response.content if hasattr(block, 'input')), None)
                    if tool_use:
                        logger.info(f"Tool parameters: {tool_use.input}")
                        result = execute_calculator_tool(tool_use.input)
                        logger.info(f"Calculator result: {result}")
                        
                        if self.validate_result(result, tool_use.input["operation"]):
                            return result
                    last_error = "Invalid result or no tool use block found"
                    logger.warning(f"Attempt {attempts + 1} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error during attempt {attempts + 1}: {last_error}")
            
            attempts += 1
            
        logger.error(f"All {self.max_retries} retry attempts failed. Last error: {last_error}")
        return {"error": f"Failed after {self.max_retries} attempts. Last error: {last_error}"}

class ConsensusRetryStrategy:
    def __init__(self, tolerance: float = 0.001, max_retries: int = 5):
        self.tolerance = tolerance
        self.max_retries = max_retries
    
    def results_agree(self, results: List[float]) -> bool:
        """Check if multiple results agree within tolerance"""
        if not results:
            return False
        return all(abs(r - results[0]) < self.tolerance for r in results)

    def call_with_consensus(self, client: Anthropic, user_input: str, calculator_tool: dict) -> Dict[str, Any]:
        """Consensus-based retry logic"""
        results = []
        attempts = 0
        
        while attempts < self.max_retries:
            try:
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    temperature=0.2 * attempts,
                    system="Use the calculator tool for all mathematical operations.",
                    messages=[{"role": "user", "content": user_input}],
                    tools=[calculator_tool]
                )
                
                if response.stop_reason == "tool_use":
                    tool_use = response.content[0]
                    result = execute_calculator_tool(tool_use.input)
                    
                    if "result" in result:
                        results.append(result["result"])
                        
                        if len(results) >= 2 and self.results_agree(results):
                            return {"result": sum(results) / len(results)}
                
            except Exception:
                pass
            
            attempts += 1
            
        return {"error": "Could not reach consensus"}

class ContextEnhancedRetryStrategy:
    def __init__(self, max_retries: int = 5):
        self.max_retries = max_retries
        self.context_templates = {
            1: (
                "This is a calculation problem. Please follow these steps:\n"
                "1. Identify the numbers involved\n"
                "2. Determine the operation needed\n"
                "3. Use the calculator tool to solve it\n"
                "Problem: "
            ),
            2: (
                "Let's solve this mathematically. Extract the numbers and operation, "
                "then use the calculator tool. Problem: "
            ),
            3: (
                "Break this down into a simple calculation using the calculator tool. "
                "Problem: "
            ),
            4: (
                "Remove the extra words and to help you solve the problem "
                "Problem: "
            ),
            5: (
                "Extract the numbers and deturman the users intent to before using the calculator tool. "
                "Problem: "
            )
        }
    
    def enhance_context(self, user_input: str, attempt_num: int) -> str:
        """Add context to the user input"""
        template = self.context_templates.get(attempt_num, "Please solve: ")
        return f"{template}{user_input}"

    def call_with_enhanced_context(self, client: Anthropic, user_input: str, calculator_tool: dict) -> Dict[str, Any]:
        """Context-enhanced retry logic"""
        attempts = 0
        while attempts < self.max_retries:
            try:
                enhanced_input = self.enhance_context(user_input, attempts + 1)
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1024,
                    temperature=0,
                    system="Always use the calculator tool for any mathematical operations. Never calculate mentally.",
                    messages=[{"role": "user", "content": enhanced_input}],
                    tools=[calculator_tool]
                )
                
                if response.stop_reason == "tool_use":
                    tool_use = response.content[0]
                    result = execute_calculator_tool(tool_use.input)
                    
                    if "result" in result:
                        return result
                
            except Exception:
                pass
            
            attempts += 1
            
        return {"error": "Failed to get valid result with enhanced context"}

class HybridRetryStrategy:
    def __init__(self, max_retries: int = 5, critical_calculation: bool = False):
        self.max_retries = max_retries
        self.critical_calculation = critical_calculation
        self.validation_strategy = ValidationRetryStrategy(max_retries)
        self.consensus_strategy = ConsensusRetryStrategy(max_retries=max_retries)
        self.context_strategy = ContextEnhancedRetryStrategy(max_retries)
    
    def execute_calculation(self, client: Anthropic, user_input: str, calculator_tool: dict) -> Dict[str, Any]:
        """Execute calculation using hybrid approach"""
        # First try with context enhancement
        result = self.context_strategy.call_with_enhanced_context(client, user_input, calculator_tool)
        if "result" in result and "error" not in result:
            return result
        
        # For critical calculations, try consensus
        if self.critical_calculation:
            result = self.consensus_strategy.call_with_consensus(client, user_input, calculator_tool)
            if "result" in result and "error" not in result:
                return result
        
        # Fall back to validation-based retry
        return self.validation_strategy.call_with_retry(client, user_input, calculator_tool)

    def format_result(self, result: Dict[str, Any], user_input: str) -> str:
        """Format the result based on context"""
        if "error" in result:
            return f"Calculation failed: {result['error']}"
        
        value = result['result']
        
        # Format based on context
        if '$' in user_input:
            return f"Result: ${value:,.2f}"
        elif 'cookies' in user_input:
            return f"Result: {value:.0f} cookies per child"
        elif 'books' in user_input:
            return f"Result: {value:.0f} books total"
        elif isinstance(value, float) and value.is_integer():
            return f"Result: {int(value)}"
        else:
            return f"Result: {value}"

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Implement rate limiting"""
        now = time.time()
        minute_ago = now - 60
        
        with self.lock:
            # Remove old requests
            self.requests = [req_time for req_time in self.requests if req_time > minute_ago]
            
            if len(self.requests) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(now)

def call_claude_with_calculator(user_input: str, critical: bool = False) -> str:
    """Main interface function for calculator"""
    try:
        rate_limiter = RateLimiter()
        retry_strategy = HybridRetryStrategy(
            max_retries=5,
            critical_calculation=critical
        )
        
        rate_limiter.wait_if_needed()
        result = retry_strategy.execute_calculation(
            client,
            user_input,
            calculator_tool
        )
        
        return retry_strategy.format_result(result, user_input)
        
    except Exception as e:
        logger.error(f"Calculation error: {str(e)}", exc_info=True)
        return f"System error: {str(e)}"

def test_calculator_functionality():
    """Test basic calculator functionality"""
    test_cases = [
        ("add", 5, 3, 8),
        ("subtract", 10, 4, 6),
        ("multiply", 6, 7, 42),
        ("divide", 20, 5, 4)
    ]
    
    for operation, op1, op2, expected in test_cases:
        try:
            result = calculator(operation, op1, op2)
            assert abs(result - expected) < 0.0001
            print(f"✓ {operation}: {op1} {operation} {op2} = {result}")
        except Exception as e:
            print(f"✗ {operation} failed: {str(e)}")

def test_retry_strategies():
    """Test retry strategies with various types of problems"""
    test_inputs = [
        "What is 1234 multiplied by 5678?",
        "Can you add 42 and 58?",
        "Divide 100 by 2",
        "If I have 145 apples and give away 37, how many do I have left?",
        "I bought 23 books and received 12 more as gifts. How many books do I have?",
        "Split 500 cookies among 20 children equally",
        "A store sold 156 items at $13 each. What was the total sales?"
    ]
    
    for input_text in test_inputs:
        print(f"\nTesting: {input_text}")
        result = call_claude_with_calculator(input_text, critical=True)
        print(f"Result: {result}")

if __name__ == "__main__":
    print("Testing Calculator Functionality:")
    test_calculator_functionality()
    
    print("\nTesting Retry Strategies:")
    test_retry_strategies()