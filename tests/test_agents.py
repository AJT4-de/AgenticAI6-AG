import pytest
from unittest.mock import MagicMock, patch

from agents.interpreter_agent import create_interpreter_agent, create_interpreter_task
from agents.writer_agent import create_writer_agent, create_writer_task
from agents.reviewer_agent import create_reviewer_agent, create_reviewer_task
from agents.illustrator_agent import create_illustrator_agent, create_illustrator_task
from agents.publisher_agent import create_publisher_agent, create_publisher_task
from crewai import Agent

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    return llm

def test_interpreter_agent_creation(mock_llm):
    agent = create_interpreter_agent(mock_llm)
    assert isinstance(agent, Agent)
    assert agent.role == "Content Strategist & Linguist"

def test_interpreter_task_creation(mock_llm):
    agent = create_interpreter_agent(mock_llm)
    task = create_interpreter_task(agent, raw_input="test", target_group="test", output_language="test")
    assert task.agent == agent
    assert task.output_pydantic is not None

def test_writer_agent_creation(mock_llm):
    agent = create_writer_agent(mock_llm)
    assert isinstance(agent, Agent)

def test_writer_task_creation(mock_llm):
    agent = create_writer_agent(mock_llm)
    task = create_writer_task(agent, raw_input="test")
    assert task.agent == agent
    assert task.output_pydantic is not None

def test_reviewer_agent_creation(mock_llm):
    agent = create_reviewer_agent(mock_llm)
    assert isinstance(agent, Agent)
    assert agent.allow_delegation is True

def test_reviewer_task_creation(mock_llm):
    agent = create_reviewer_agent(mock_llm)
    task = create_reviewer_task(agent)
    assert task.agent == agent

def test_illustrator_agent_creation(mock_llm):
    agent = create_illustrator_agent(mock_llm)
    assert isinstance(agent, Agent)

def test_illustrator_task_creation(mock_llm):
    agent = create_illustrator_agent(mock_llm)
    task = create_illustrator_task(agent)
    assert task.agent == agent

def test_publisher_agent_creation(mock_llm):
    agent = create_publisher_agent(mock_llm)
    assert isinstance(agent, Agent)

def test_publisher_task_creation(mock_llm):
    agent = create_publisher_agent(mock_llm)
    task = create_publisher_task(agent)
    assert task.agent == agent
