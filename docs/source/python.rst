=======
Python
=======

Python is a widely-used programming language for data science and machine learning. It is known for its simplicity and readability, making it a great choice for both beginners and experienced programmers.    

Docstrings
----------
Docstrings are a way to document your code. They are written in triple quotes and can be used to describe the purpose of a function, class, or module. Docstrings can be accessed using the `__doc__` attribute.

Here is an example of a class and a function:

.. code-block:: python
  
  # Example of a class
  class MyClass:
    """
    This is a simple example class.

    Attributes:
        attr1 (int): An integer attribute.
        attr2 (str): A string attribute.
    """

    def __init__(self, attr1: int, attr2: str):
        """
        Initialize the MyClass instance.

        Args:
            attr1 (int): The first attribute.
            attr2 (str): The second attribute.
        """
        self.attr1 = attr1
        self.attr2 = attr2

  # Example of a function
  def add(a: int, b: int) -> int:
      """
      Add two numbers together.

      Args:
          a (int): The first number.
          b (int): The second number.

      Returns:
          int: The sum of the two numbers.
      """
      return a + b

Decorators
-----------
Decorators is a way to modify the behavior of a function or class. They are defined using the `@` symbol and can be used to add functionality to existing code without modifying it.

A simple expample of a decorator is:

.. code-block:: python

  def my_decorator(func):
      def wrapper():
          print("Something is happening before the function is called.")
          func()
          print("Something is happening after the function is called.")
      return wrapper

  @my_decorator
  def say_hello():
      print("Hello!")

  say_hello()

Using decorator to implement authentication:

.. code-block:: python

  def authenticate(func):
      def wrapper(*args, **kwargs):
          if user.is_authenticated:
              return func(*args, **kwargs)
          else:
              raise Exception("User must be authenticated to call this function.")
      return wrapper

  @authenticate
  def sensitive_function():
      print("This is a sensitive function.")

  sensitive_function()

Hooks
-----
Hooks refer to functions that can be overridden or extended to customize the behavior of a larger system or library.

Here is an example of using hooks to extend behaviors:

.. code-block:: python

  import time
  
  class Programmer(object):
    def __init__(self, name, hook=None):
        self.name = name
        self.hooks_func = hook
        self.now_date = time.strftime("%Y-%m-%d")

    def get_to_eat(self):
        print(f"{self.name} - {self.now_date}: eat.")

    def go_to_code(self):
        print(f"{self.name} - {self.now_date}: code.")

    def go_to_sleep(self):
        print(f"{self.name} - {self.now_date}: sleep.")

    def everyday(self):
        self.get_to_eat()
        self.go_to_code()
        self.go_to_sleep()
        # check the register_hook(hooked or unhooked)
        # hooked
        if self.hooks_func is not None:
            self.hooks_func(self.name)

  
  def play_game(name):
    now_date = time.strftime("%Y-%m-%d")
    print(f"{name} - {now_date}: play game.")


  def shopping(name):
    now_date = time.strftime("%Y-%m-%d")
    print(f"{name} - {now_date}: shopping.")


  if __name__ == "__main__":
    tom = Programmer("Tom", hook=play_game)
    jerry = Programmer("Jerry", hook=shopping)
    spike = Programmer("Spike")
    tom.everyday()
    jerry.everyday()
    spike.everyday()


Typing hints
------------
Typing hints are a way to indicate the expected types of function arguments and return values. This can help with code readability and can also be used by static type checkers to catch potential errors.

Here is an example of how to use typing hints:

.. code-block:: python

    def process_data(data: list[str]) -> dict[str, int]:
        """
        Process a list of strings and return a dictionary with the count of each string.

        Args:
            data (list[str]): A list of strings.

        Returns:
            dict[str, int]: A dictionary with the count of each string.
        """
        result = {}
        for item in data:
            result[item] = result.get(item, 0) + 1
        return result



Logging
--------
Logging is a way to track events that happen when your code runs. The logging module in Python allows you to log messages at different severity levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

Here is an example of how to use the logging module:

.. code-block:: python

  import logging
  # Configure logging
  logging.basicConfig(level=logging.DEBUG)

  # Example of logging usage
  logging.debug("This is a debug message.")
  logging.info("This is an info message.")
  logging.warning("This is a warning message.")
  logging.error("This is an error message.")
  logging.critical("This is a critical message.")


Virtual environments
---------------------
Virtual environments are a way to create isolated Python environments. This is useful for managing dependencies and avoiding conflicts between different projects.

Here is an example of using anconda to create and manage virtual environments:

.. code-block:: bash

  # Create a new virtual environment
  conda create -n myenv python=3.8

  # Activate the virtual environment
  conda activate myenv

  # Install packages in the virtual environment
  conda install numpy pandas matplotlib

  # List all virtual environments
  conda env list

  # Deactivate the virtual environment
  conda deactivate

  # Remove the virtual environment
  conda remove -n myenv --all

Debugging
-------------
Debugging is the process of finding and fixing errors in your code. Python provides several tools for debugging, including the built-in `pdb` module and IDEs like PyCharm and VSCode.

Here is an example of visualizing python execution with `VizTracer <https://github.com/gaogaotiantian/viztracer>`_:

.. code-block:: bash

  # Install VizTracer
  pip install viztracer

  # Run your script with VizTracer
  viztracer my_script.py

  # Open the generated JSON report
  vizviewer result.json


Testing
--------
Testing is the process of verifying that your code works as expected. Python provides several libraries for testing, including `unittest`, `pytest`, and `doctest`.

Here is an example of using `pytest` to test a function:

.. code-block:: python

    # test_my_module.py
    import pytest
    from my_module import add
    
    def test_add():
        assert add(1, 2) == 3
        assert add(-1, 1) == 0
        assert add(0, 0) == 0
    
    if __name__ == "__main__":
        pytest.main()

Packaging
---------
Packaging is the process of creating a distributable version of your code. This can be done using tools like `setuptools` and `wheel`. Packaging your code makes it easier to share with others and to install in different environments.

Here is an example of packaging projects with `setup.py`:

.. code-block:: bash

    # file structure
    my_package/
        ├── __init__.py
        ├── my_module.py
        ├── my_script.py
        ├── requirements.txt
        └── setup.py

.. code-block:: python

    # setup.py
    from setuptools import setup, find_packages
    
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    setup(
        name="my_package",
        version="0.1",
        packages=find_packages(),
        install_requires=requirements,
    )

.. code-block:: bash

    # Build the package
    python setup.py sdist bdist_wheel

    # Install the package
    pip install my_package-0.1-py3-none-any.whl

    # Uninstall the package
    pip uninstall my_package

.. code-block:: python

    import mypackage.a as a
    import mypackage.b as b

