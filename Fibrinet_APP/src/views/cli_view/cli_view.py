from src.managers.view.view_strategy import ViewStrategy
from src.models.exceptions import StateTransitionError, UnsupportedFileTypeError, InvalidInputDataError
from utils.logger.logger import Logger
import os
import platform
import shlex
from typing import List, Dict, Any, Optional

class CommandLineView(ViewStrategy):
    """Command-line interface for FibriNet."""
    
    def __init__(self, controller):
        self.controller = controller 
        self.running = True
        self.command_history = []
        self.max_history = 50
        
        # Command catalog
        self.commands = {
            'help': {
                'description': 'Show available commands and their usage',
                'usage': 'help [command]',
                'args': 'optional',
                'examples': ['help', 'help input_network']
            },
            'exit': {
                'description': 'Exit the CLI',
                'usage': 'exit',
                'args': 'none',
                'examples': ['exit']
            },
            'quit': {
                'description': 'Exit the CLI',
                'usage': 'quit',
                'args': 'none',
                'examples': ['quit']
            },
            'clear': {
                'description': 'Clear the terminal screen',
                'usage': 'clear',
                'args': 'none',
                'examples': ['clear']
            },
            'status': {
                'description': 'Show current system status and network information',
                'usage': 'status',
                'args': 'none',
                'examples': ['status']
            },
            'input_network': {
                'description': 'Load a network from an Excel file',
                'usage': 'input_network <file_path>',
                'args': 'required',
                'examples': ['input_network test/input_data/INPUT TESTS/TestNetwork.xlsx']
            },
            'set_degradation_engine': {
                'description': 'Set the degradation engine strategy',
                'usage': 'set_degradation_engine <strategy>',
                'args': 'required',
                'examples': ['set_degradation_engine nophysics', 'set_degradation_engine twodimensionalspringforcedegradationenginewithoutbiomechanics']
            },
            'degrade_node': {
                'description': 'Degrade a specific node in the network',
                'usage': 'degrade_node <node_id>',
                'args': 'required',
                'examples': ['degrade_node 1']
            },
            'degrade_edge': {
                'description': 'Degrade a specific edge in the network',
                'usage': 'degrade_edge <edge_id>',
                'args': 'required',
                'examples': ['degrade_edge 1']
            },
            'undo': {
                'description': 'Undo the last degradation operation',
                'usage': 'undo',
                'args': 'none',
                'examples': ['undo']
            },
            'redo': {
                'description': 'Redo the last undone degradation operation',
                'usage': 'redo',
                'args': 'none',
                'examples': ['redo']
            },
            'relax': {
                'description': 'Relax the network to find equilibrium state',
                'usage': 'relax',
                'args': 'none',
                'examples': ['relax']
            },
            'export': {
                'description': 'Export network data and/or images',
                'usage': 'export <data_strategy> <image_strategy> <folder_path>',
                'args': 'required',
                'examples': ['export excel_data_export_strategy none ./exports', 'export none png_image_export_strategy ./exports']
            },
            'configure_logger': {
                'description': 'Configure logging settings',
                'usage': 'configure_logger <enable|disable> [file_location]',
                'args': 'required',
                'examples': ['configure_logger enable', 'configure_logger disable', 'configure_logger enable ./logs.txt']
            },
            'switch_view': {
                'description': 'Switch to a different view (e.g., GUI)',
                'usage': 'switch_view <view_type>',
                'args': 'required',
                'examples': ['switch_view tkinter']
            },
            'reset': {
                'description': 'Reset the current network and clear history',
                'usage': 'reset',
                'args': 'none',
                'examples': ['reset']
            },
            'history': {
                'description': 'Show command history',
                'usage': 'history [number]',
                'args': 'optional',
                'examples': ['history', 'history 10']
            },
            'spring_constant': {
                'description': 'Get or set the spring stiffness constant',
                'usage': 'spring_constant [new_value]',
                'args': 'optional',
                'examples': ['spring_constant', 'spring_constant 2.5']
            },
            'reset_spring_constant': {
                'description': 'Reset spring constant to original value from input file',
                'usage': 'reset_spring_constant',
                'args': 'none',
                'examples': ['reset_spring_constant']
            }
        }

    def clear_view(self):
        """Clear the terminal screen."""
        Logger.log("start clear_view()")
        system = platform.system()
        if system == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
        Logger.log("end clear_view()")

    def start_view(self):
        """Start the CLI."""
        Logger.log("start start_view()")
        self.clear_view()
        self._print_welcome()
        self.run()
        Logger.log("end start_view()")

    def stop_view(self):
        """Stop the CLI."""
        Logger.log("start stop_view()")
        self.running = False
        print("\n>>> FibriNet CLI stopped. Goodbye!\n")
        Logger.log("end stop_view()")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 60)
        print("           FIBRINET COMMAND LINE INTERFACE")
        print("=" * 60)
        print("Network Modeling and Degradation Analysis Tool")
        print("Type 'help' to see available commands")
        print("Type 'exit' or 'quit' to exit")
        print("=" * 60)

    def run(self):
        """Main command loop."""
        Logger.log("start run()")
        
        while self.running:
            try:
                command = input("\nFibriNet> ").strip()
                if not command:
                    continue
                self._add_to_history(command)
                self._process_command(command)
            except KeyboardInterrupt:
                print("\n>>> Use 'exit' or 'quit' to exit the application.")
            except EOFError:
                print("\n>>> End of input. Exiting...")
                self.stop_view()
                break
            except Exception as ex:
                print(f">>> Unexpected error: {ex}")
                Logger.log(f"Unexpected error in CLI: {ex}", Logger.LogPriority.ERROR)
        
        Logger.log("end run()")

    def _add_to_history(self, command: str):
        """Track command history."""
        if command not in self.command_history:
            self.command_history.append(command)
            if len(self.command_history) > self.max_history:
                self.command_history.pop(0)

    def _process_command(self, command: str):
        """Parse and dispatch a command."""
        try:
            parts = shlex.split(command)
            if not parts:
                return
            cmd = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            Logger.log(f"Processing command: {cmd} with args: {args}")
            self._execute_command(cmd, args)
        except Exception as ex:
            print(f">>> Error processing command: {ex}")
            Logger.log(f"Error processing command '{command}': {ex}", Logger.LogPriority.ERROR)

    def _execute_command(self, cmd: str, args: List[str]):
        """Execute a command."""
        if cmd in ['exit', 'quit']:
            self.stop_view(); return
        elif cmd == 'clear':
            self.clear_view(); self._print_welcome(); return
        elif cmd == 'help':
            self._handle_help(args); return
        elif cmd == 'status':
            self._handle_status(); return
        elif cmd == 'history':
            self._handle_history(args); return
        elif cmd == 'input_network':
            self._handle_input_network(args); return
        elif cmd == 'set_degradation_engine':
            self._handle_set_degradation_engine(args); return
        elif cmd == 'degrade_node':
            self._handle_degrade_node(args); return
        elif cmd == 'degrade_edge':
            self._handle_degrade_edge(args); return
        elif cmd == 'undo':
            self._handle_undo(); return
        elif cmd == 'redo':
            self._handle_redo(); return
        elif cmd == 'relax':
            self._handle_relax(); return
        elif cmd == 'export':
            self._handle_export(args); return
        elif cmd == 'configure_logger':
            self._handle_configure_logger(args); return
        elif cmd == 'switch_view':
            self._handle_switch_view(args); return
        elif cmd == 'reset':
            self._handle_reset(); return
        elif cmd == 'spring_constant':
            self._handle_spring_constant(args); return
        elif cmd == 'reset_spring_constant':
            self._handle_reset_spring_constant(); return
        else:
            print(f">>> Unknown command: '{cmd}'")
            print(">>> Type 'help' to see available commands")
            return

    def _handle_help(self, args: List[str]):
        """Help command."""
        if not args:
            self._show_general_help()
        else:
            self._show_command_help(args[0])

    def _show_general_help(self):
        """List commands."""
        print("\n" + "=" * 60)
        print("                        FIBRINET CLI HELP")
        print("=" * 60)
        print("Available commands:\n")
        for cmd, info in self.commands.items():
            print(f"  {cmd:<20} - {info['description']}")
        print("\nFor detailed help on a specific command, type: help <command>")
        print("=" * 60)

    def _show_command_help(self, command: str):
        """Show command details."""
        if command not in self.commands:
            print(f">>> Unknown command: '{command}'"); return
        info = self.commands[command]
        print(f"\nCommand: {command}")
        print(f"Description: {info['description']}")
        print(f"Usage: {info['usage']}")
        print(f"Arguments: {info['args']}")
        print("Examples:")
        for example in info['examples']:
            print(f"  {example}")

    def _handle_status(self):
        """Status command."""
        print("\n" + "=" * 40)
        print("           SYSTEM STATUS")
        print("=" * 40)
        network_loaded = self.controller.system_state.network_loaded
        print(f"Network Loaded: {'Yes' if network_loaded else 'No'}")
        if network_loaded:
            network = self.controller.network_manager.get_network()
            if network:
                print(f"Number of Nodes: {len(network.get_nodes())}")
                print(f"Number of Edges: {len(network.get_edges())}")
                print(f"Network Type: {type(network).__name__}")
                state_manager = self.controller.network_manager.state_manager
                print(f"History States: {len(state_manager.network_state_history)}")
                print(f"Current State Index: {state_manager.current_network_state_index}")
                print(f"Undo Available: {'No' if state_manager.undo_disabled else 'Yes'}")
                print(f"Redo Available: {'No' if state_manager.redo_disabled else 'Yes'}")
                print(f"Export Available: {'No' if state_manager.export_disabled else 'Yes'}")
        engine = self.controller.network_manager.degradation_engine_strategy
        print(f"Degradation Engine: {type(engine).__name__}")
        if network_loaded:
            current_spring = self.controller.get_spring_constant()
            original_spring = self.controller.get_original_spring_constant()
            print(f"Current Spring Constant: {current_spring}")
            print(f"Original Spring Constant: {original_spring}")
        print("=" * 40)

    def _handle_history(self, args: List[str]):
        """History command."""
        try:
            if args:
                num = int(args[0])
                if num <= 0:
                    print(">>> Number must be positive"); return
                history = self.command_history[-num:]
            else:
                history = self.command_history[-10:]
            if not history:
                print(">>> No command history available"); return
            print("\nCommand History:")
            for i, cmd in enumerate(history, 1):
                print(f"  {i:2d}. {cmd}")
        except ValueError:
            print(">>> Invalid number format")

    def _handle_input_network(self, args: List[str]):
        """Load a network."""
        if not args:
            print(">>> Error: File path required")
            print(">>> Usage: input_network <file_path>"); return
        file_path = args[0]
        try:
            self.controller.input_network(file_path)
            print(f">>> Network loaded successfully from: {file_path}")
            network = self.controller.network_manager.get_network()
            if network:
                print(f">>> Network contains {len(network.get_nodes())} nodes and {len(network.get_edges())} edges")
        except FileNotFoundError:
            print(f">>> Error: File not found: {file_path}")
        except UnsupportedFileTypeError as ex:
            print(f">>> Error: {ex}")
        except InvalidInputDataError as ex:
            print(f">>> Error: Invalid input data - {ex}")
        except Exception as ex:
            print(f">>> Error loading network: {ex}")

    def _handle_set_degradation_engine(self, args: List[str]):
        """Set degradation engine."""
        if not args:
            print(">>> Error: Strategy name required")
            print(">>> Usage: set_degradation_engine <strategy>")
            print(">>> Available strategies: nophysics, twodimensionalspringforcedegradationenginewithoutbiomechanics"); return
        strategy = args[0]
        try:
            self.controller.set_degradation_engine_strategy(strategy)
            print(f">>> Degradation engine set to: {strategy}")
        except Exception as ex:
            print(f">>> Error setting degradation engine: {ex}")

    def _handle_degrade_node(self, args: List[str]):
        """Degrade a node."""
        if not args:
            print(">>> Error: Node ID required")
            print(">>> Usage: degrade_node <node_id>"); return
        try:
            node_id = int(args[0])
            self.controller.degrade_node(node_id)
            print(f">>> Node {node_id} degraded successfully")
        except ValueError:
            print(">>> Error: Node ID must be a number")
        except StateTransitionError:
            print(">>> Error: Network must be loaded before degrading nodes")
        except Exception as ex:
            print(f">>> Error degrading node: {ex}")

    def _handle_degrade_edge(self, args: List[str]):
        """Degrade an edge."""
        if not args:
            print(">>> Error: Edge ID required")
            print(">>> Usage: degrade_edge <edge_id>"); return
        try:
            edge_id = int(args[0])
            self.controller.degrade_edge(edge_id)
            print(f">>> Edge {edge_id} degraded successfully")
        except ValueError:
            print(">>> Error: Edge ID must be a number")
        except StateTransitionError:
            print(">>> Error: Network must be loaded before degrading edges")
        except Exception as ex:
            print(f">>> Error degrading edge: {ex}")

    def _handle_undo(self):
        """Undo last change."""
        try:
            self.controller.undo_degradation()
            print(">>> Last degradation undone successfully")
        except StateTransitionError:
            print(">>> Error: Network must be loaded before undoing")
        except Exception as ex:
            print(f">>> Error undoing degradation: {ex}")

    def _handle_redo(self):
        """Redo last undone change."""
        try:
            self.controller.redo_degradation()
            print(">>> Last undone degradation redone successfully")
        except StateTransitionError:
            print(">>> Error: Network must be loaded before redoing")
        except Exception as ex:
            print(f">>> Error redoing degradation: {ex}")

    def _handle_relax(self):
        """Relax the network."""
        try:
            self.controller.network_manager.relax_network()
            print(">>> Network relaxed successfully")
        except Exception as ex:
            print(f">>> Error relaxing network: {ex}")

    def _handle_export(self, args: List[str]):
        """Export results."""
        if len(args) != 3:
            print(">>> Error: Three arguments required")
            print(">>> Usage: export <data_strategy> <image_strategy> <folder_path>")
            print(">>> Examples:")
            print(">>>   export excel_data_export_strategy none ./exports")
            print(">>>   export none png_image_export_strategy ./exports"); return
        data_strategy, image_strategy, folder_path = args
        export_request = f"export_request {data_strategy} {image_strategy} {folder_path}"
        try:
            self.controller.export_data(export_request)
            print(f">>> Export completed successfully to: {folder_path}")
        except StateTransitionError:
            print(">>> Error: Cannot export - network not loaded or export conditions not met")
        except Exception as ex:
            print(f">>> Error during export: {ex}")

    def _handle_configure_logger(self, args: List[str]):
        """Configure logging."""
        if not args:
            print(">>> Error: Logger state required")
            print(">>> Usage: configure_logger <enable|disable> [file_location]"); return
        state = args[0].lower()
        if state not in ['enable', 'disable']:
            print(">>> Error: Logger state must be 'enable' or 'disable'"); return
        enabled = state == 'enable'
        kwargs = {}
        if len(args) > 1 and enabled:
            kwargs['storage_strategy'] = 'file'
            kwargs['file_location'] = args[1]
        try:
            self.controller.configure_Logger(enabled, **kwargs)
            print(f">>> Logger {state}d successfully")
        except Exception as ex:
            print(f">>> Error configuring logger: {ex}")

    def _handle_switch_view(self, args: List[str]):
        """Switch view."""
        if not args:
            print(">>> Error: View type required")
            print(">>> Usage: switch_view <view_type>")
            print(">>> Available views: tkinter"); return
        view_type = args[0]
        try:
            self.controller.initiate_view(view_type)
            print(f">>> Switched to {view_type} view")
        except ValueError as ex:
            print(f">>> Error: {ex}")
        except Exception as ex:
            print(f">>> Error switching view: {ex}")

    def _handle_reset(self):
        """Reset network and history."""
        try:
            self.controller.network_manager.reset_network_and_state()
            self.controller.system_state.network_loaded = False
            print(">>> Network reset successfully")
        except Exception as ex:
            print(f">>> Error resetting network: {ex}")

    def _handle_spring_constant(self, args: List[str]):
        """Get or set spring constant."""
        if not self.controller.system_state.network_loaded:
            print(">>> Error: No network loaded"); return
        if not args:
            current = self.controller.get_spring_constant()
            original = self.controller.get_original_spring_constant()
            print(f">>> Current spring constant: {current}")
            print(f">>> Original spring constant: {original}")
        else:
            try:
                new_value = float(args[0])
                self.controller.set_spring_constant(new_value)
                print(f">>> Spring constant updated to: {new_value}")
                print(">>> Network has been relaxed with new physics parameters")
            except ValueError:
                print(">>> Error: Spring constant must be a number")
            except StateTransitionError as ex:
                print(f">>> Error: {ex}")
            except Exception as ex:
                print(f">>> Error updating spring constant: {ex}")

    def _handle_reset_spring_constant(self):
        """Reset spring constant to original value."""
        if not self.controller.system_state.network_loaded:
            print(">>> Error: No network loaded"); return
        try:
            original = self.controller.get_original_spring_constant()
            if original is not None:
                self.controller.reset_spring_constant()
                print(f">>> Spring constant reset to original value: {original}")
            else:
                print(">>> Error: No original spring constant found")
        except StateTransitionError as ex:
            print(f">>> Error: {ex}")
        except Exception as ex:
            print(f">>> Error resetting spring constant: {ex}")
