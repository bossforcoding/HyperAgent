import os
import warnings
from pathlib import Path
from typing import Optional

from autogen import UserProxyAgent

from hyperagent.agents.plan_seeking import load_agent_navigator, load_agent_editor, load_agent_executor, \
    load_summarizer, load_agent_planner, load_manager
from hyperagent.build import initialize_tools
from hyperagent.constants import DEFAULT_VERBOSE_LEVEL, DEFAULT_LLM_CONFIGS, DEFAULT_TRAJECTORIES_PATH, \
    DEFAULT_IMAGE_NAME, DEFAULT_PATCHES_DIR
from hyperagent.prompts.editor import system_edit
from hyperagent.prompts.executor import system_exec
from hyperagent.prompts.navigator import system_nav
from hyperagent.prompts.planner import system_plan
from hyperagent.utils import clone_repo, check_local_or_remote, setup_logger

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = setup_logger()


def Setup(
        repo_path: str,
        commit: str = "HEAD",
        language: str = "python",
        clone_dir: str = "data/repos",
        save_trajectories_path: Optional[str] = DEFAULT_TRAJECTORIES_PATH,
        db_path: Optional[str] = None,
        index_path: Optional[str] = "data/indexes",
        llm_configs: Optional[dict] = None,
        image_name: Optional[str] = DEFAULT_IMAGE_NAME,
        force_local: bool = False,
):
    if not os.path.exists(DEFAULT_PATCHES_DIR):
        os.makedirs(DEFAULT_PATCHES_DIR)

    # Gestione path locale
    if force_local or os.path.exists(repo_path):
        # Usa direttamente il path locale
        repo_dir = Path(repo_path).resolve()
        if not repo_dir.exists():
            raise ValueError(f"Local path does not exist: {repo_dir}")

        logger.info(f"Using local repository at: {repo_dir}")
        repo_dir = str(repo_dir)

        # Se è un repo git e viene specificato un commit, fai checkout
        if commit != "HEAD" and (Path(repo_dir) / ".git").exists():
            try:
                import git
                repo = git.Repo(repo_dir)
                repo.git.checkout(commit)
                logger.info(f"Checked out commit: {commit}")
            except Exception as e:
                logger.warning(f"Could not checkout commit {commit}: {e}")
    else:
        # Comportamento originale: clona il repository
        gh_token = os.environ.get("GITHUB_TOKEN", None)
        is_local, repo_path = check_local_or_remote(repo_path)

        if is_local:
            repo_dir = repo_path
        else:
            repo_dir = clone_repo(repo_path, commit, clone_dir, gh_token, logger)

    # Assicurati che il path sia assoluto
    if not os.path.isabs(repo_dir):
        repo_dir = os.path.join(os.getcwd(), repo_dir)

    # if save_trajectories_path and not os.path.exists(save_trajectories_path):
    #     print(save_trajectories_path)
    #     os.makedirs(save_trajectories_path)

    # Set up the tool kernel
    jupyter_executor, docker_executor = initialize_tools(repo_dir, db_path, index_path, language, image_name)
    logger.info("Initialized tools")

    # Set up the navigator, executor and generator agent (the system)
    summarizer = load_summarizer()

    user_proxy = UserProxyAgent(
        name="Admin",
        system_message="A human admin. Interact with the planner to discuss the plan to resolve a codebase-related query.",
        human_input_mode="ALWAYS",
        code_execution_config=False,
        default_auto_reply="",
        max_consecutive_auto_reply=0
    )

    navigator = load_agent_navigator(
        llm_configs["nav"],
        jupyter_executor,
        system_nav,
        summarizer
    )

    editor = load_agent_editor(
        llm_configs["edit"],
        jupyter_executor,
        system_edit,
        repo_dir
    )

    executor = load_agent_executor(
        llm_configs["exec"],
        docker_executor,
        system_exec,
        summarizer
    )

    planner = load_agent_planner(
        system_plan,
        llm_configs["plan"]
    )

    manager = load_manager(
        user_proxy=user_proxy,
        planner=planner,
        navigator=navigator,
        editor=editor,
        executor=executor,
        llm_config=llm_configs
    )

    return manager, user_proxy, repo_dir


class HyperAgent:
    def __init__(
            self,
            repo_path,
            commit="HEAD",  # Default a HEAD invece di "str"
            language="python",
            clone_dir="data/repos",
            save_trajectories_path=DEFAULT_TRAJECTORIES_PATH,
            llm_configs=DEFAULT_LLM_CONFIGS,
            verbose=DEFAULT_VERBOSE_LEVEL,
            image_name=DEFAULT_IMAGE_NAME,
            force_local=True  # Nuovo parametro, default True per usare sempre path locali
    ):
        self.repo_path = repo_path
        self.language = language
        self.force_local = force_local

        # Log se stiamo usando modalità locale
        if force_local or os.path.exists(repo_path):
            logger.info(f"Initializing HyperAgent in local mode for: {repo_path}")

        self.system, self.user_proxy, repo_dir = Setup(
            self.repo_path,
            commit,
            language=language,
            clone_dir=clone_dir,
            save_trajectories_path=save_trajectories_path,
            llm_configs=llm_configs,
            image_name=image_name,
            force_local=force_local
        )

        self.repo_dir = repo_dir
        self.verbose = verbose

    def query_codebase(self, query):
        return self.user_proxy.initiate_chat(
            self.system, message=query, silent=self.verbose
        )

    # Metodo helper per verificare lo stato
    def get_repo_info(self):
        """Ritorna informazioni sul repository corrente"""
        return {
            "path": self.repo_dir,
            "language": self.language,
            "is_local": self.force_local or os.path.exists(self.repo_path),
            "exists": os.path.exists(self.repo_dir)
        }
