# src/api/dependencies/orchestrator_dep.py
from src.agents.orchestrator.orchestrator import MultiAgentOrchestrator
from src.agents.orchestrator.host_agent import HostAgent
from src.agents.doubts.active_recall import ActiveRecallDrillAgent
from src.agents.doubts.misconception import MisconceptionCorrectorAgent
from src.agents.doubts.worked_solution import WorkedSolutionProviderAgent
from src.agents.generation.flashcard import FlashcardGeneratorAgent
from src.agents.generation.summary import SummaryGeneratorAgent
from src.agents.generation.practice_questions import PracticeQuestionGeneratorAgent
from src.agents.generation.study_planner import StudyPlannerAgent
from src.agents.problem_solver.comparison import ComparisonAgent
from src.agents.problem_solver.concept_solver import ConceptExplainerAgent
from src.agents.problem_solver.problem_solver import ProblemSolverAgent

# Singleton instance of the orchestrator
_orchestrator_instance = None

def get_orchestrator() -> MultiAgentOrchestrator:
    """
    Dependency injection provider returning a pre-configured MultiAgentOrchestrator.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        orchestrator = MultiAgentOrchestrator()
        
        # Register all academic specialized agents
        orchestrator.register_agent(HostAgent())
        orchestrator.register_agent(ActiveRecallDrillAgent())
        orchestrator.register_agent(MisconceptionCorrectorAgent())
        orchestrator.register_agent(WorkedSolutionProviderAgent())
        orchestrator.register_agent(FlashcardGeneratorAgent())
        orchestrator.register_agent(SummaryGeneratorAgent())
        orchestrator.register_agent(PracticeQuestionGeneratorAgent())
        orchestrator.register_agent(StudyPlannerAgent())
        orchestrator.register_agent(ComparisonAgent())
        orchestrator.register_agent(ConceptExplainerAgent())
        orchestrator.register_agent(ProblemSolverAgent())
        
        _orchestrator_instance = orchestrator
        
    return _orchestrator_instance
