"""Setup module for openrocketengine."""
from distutils.core import setup


setup(
    name="openrocketengine",
    version="0.0.1dev",
    description="A collection of tools used for high level rocket engine design.",
    url=["https://github.com/cmflannery/openrocketengine"],
    packages=['openrocketengine', 'openrocketengine/core'],
    install_requires=["numpy", "matplotlib", "pathlib"],
    entry_points={"console_scripts": []},
)
