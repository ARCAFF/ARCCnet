Setting up a Development Environment
====================================

To set up a development environment for ``arccnet``, follow these steps:

1. **Fork the Repository**: Start by forking the ``arccnet`` repository on GitHub. Forking creates a copy of the repository under your GitHub account, allowing you to freely make changes without affecting the original project.

2. **Creating a Conda Environment (Optional)**: If you prefer to use Anaconda (herein conda) for managing your development environment, you can create a conda environment specifically for ``arccnet``:

   - **Install Conda**: If you haven't installed conda already, follow the installation instructions for your operating system from the official documentation.

   - **Create a Conda Environment**: Use the following command to create a new conda environment for ``arccnet``, replacing `<environment_name>` with a name of your choice:

     .. code-block:: bash

        $ conda create -n <environment_name> python=3.9

   - **Activate the Environment**: Activate the conda environment using:

     .. code-block:: bash

        $ conda activate <environment_name>

     From now on, all the commands you run and packages you install will be within this environment.

3. **Clone the Repository**: After forking and setting up the conda environment, clone the forked repository to your local machine, and install ``arccnet`` in an editable mode using the following commands:

   .. code-block:: bash

      $ git clone https://github.com/<username>/ARCCnet.git
      $ cd ARCCnet
      $ pip install -e .[dev]

   If you are using `zsh` instead of `bash`, the last line should be modified to:

   .. code-block:: zsh

      $ pip install -e '.[dev]'

4. **Creating a New Branch**: To work on a new feature or make changes to the project, it is recommended to create a new branch. This helps isolate your changes and makes it easier to manage multiple contributions. Use the following command to create a new branch, replacing `<name_of_branch>` with an appropriate name for your branch:

   .. code-block:: bash

      $ git checkout -b <name_of_branch>

5. **Committing Changes**: After making the desired changes to the codebase, you need to commit them to the branch. This creates a checkpoint for your changes. Use the following commands to commit your changes:

   .. code-block:: bash

      $ git add .
      $ git commit -m "Your commit message"

6. **Running Tests** (Optional): Running tests, and generate a html report

   .. code-block:: bash

      $ pytest --cov=arccnet tests/
      $ coverage html

6. **Pushing the Branch**: To push your branch to the remote repository on GitHub, use the following command:

   .. code-block:: bash

      $ git push origin <branch_name>

7. **Opening a Pull Request**: Once your changes are committed and pushed to your forked repository, you can open a pull request (PR) against the main repository.

   1. Visit the original repository on GitHub
   2. Click on the prompt to create a new pull request

Congratulations! You have successfully set up your development environment, created a conda environment (if desired), cloned the repository, created a new branch, committed your changes, pushed the branch, and opened a pull request against the main repository.
