"""Explore langchain_core package structure to help with auto-completion."""
import pkgutil
import inspect
import langchain_core

def explore_package(package, prefix=''):
    """Explore a package and print its contents."""
    print(f"\n{prefix}Package: {package.__name__}")
    
    # Print modules
    print(f"{prefix}Modules:")
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        print(f"{prefix}  - {name}")
        if ispkg:
            try:
                module = __import__(name, fromlist=['*'])
                explore_package(module, prefix + '    ')
            except ImportError:
                print(f"{prefix}    (Could not import this subpackage)")
    
    # If it's a module, print its members
    if not hasattr(package, '__path__'):
        print(f"\n{prefix}Members of {package.__name__}:")
        for name, obj in inspect.getmembers(package):
            if not name.startswith('_'):  # Skip private members
                print(f"{prefix}  - {name} ({type(obj).__name__})")

if __name__ == "__main__":
    print("LangChain Core Version:", langchain_core.__version__)
    
    # Explore language_models module specifically (what you're trying to import)
    try:
        import langchain_core.language_models
        print("\nClasses in langchain_core.language_models:")
        for name, obj in inspect.getmembers(langchain_core.language_models):
            if inspect.isclass(obj) and not name.startswith('_'):
                print(f"  - {name}")
        
        # Print specifically about BaseChatModel
        if hasattr(langchain_core.language_models, 'BaseChatModel'):
            print("\nBaseChatModel details:")
            print("  - Module:", langchain_core.language_models.BaseChatModel.__module__)
            print("  - Doc:", langchain_core.language_models.BaseChatModel.__doc__.split('\n')[0] if langchain_core.language_models.BaseChatModel.__doc__ else "No docstring")
    except ImportError as e:
        print(f"Error importing langchain_core.language_models: {e}")
    
    # Show overall package structure
    explore_package(langchain_core)
