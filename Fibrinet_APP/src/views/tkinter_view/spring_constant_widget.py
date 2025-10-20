import tkinter as tk
from tkinter import messagebox
from utils.logger.logger import Logger

class SpringConstantWidget:
    """GUI widget to view/update/reset the spring stiffness constant."""
    
    def __init__(self, parent, controller):
        """
        Initialize the spring constant widget.
        
        Args:
            parent: The parent widget (should have BG_COLOR, FG_COLOR, SUBHEADING_FONT attributes)
            controller: The system controller instance
        """
        self.parent = parent
        self.controller = controller
        self.create_widgets()
    
    def create_widgets(self):
        """Create and layout controls."""
        Logger.log("Creating spring constant widget")
        
        # Verify parent has required attributes
        if not hasattr(self.parent, 'root'):
            Logger.log("Parent missing root attribute", Logger.LogPriority.ERROR)
            raise AttributeError("Parent must have root attribute (tkinter root window)")
        if not hasattr(self.parent, 'BG_COLOR'):
            Logger.log("Parent missing BG_COLOR attribute", Logger.LogPriority.ERROR)
            raise AttributeError("Parent must have BG_COLOR attribute")
        if not hasattr(self.parent, 'FG_COLOR'):
            Logger.log("Parent missing FG_COLOR attribute", Logger.LogPriority.ERROR)
            raise AttributeError("Parent must have FG_COLOR attribute")
        if not hasattr(self.parent, 'SUBHEADING_FONT'):
            Logger.log("Parent missing SUBHEADING_FONT attribute", Logger.LogPriority.ERROR)
            raise AttributeError("Parent must have SUBHEADING_FONT attribute")
        
        # Container
        self.frame = tk.Frame(self.parent.root, bg=self.parent.BG_COLOR)
        
        # Label
        self.label = tk.Label(
            self.frame, 
            text="Spring Stiffness Constant:", 
            font=self.parent.SUBHEADING_FONT, 
            bg=self.parent.BG_COLOR, 
            fg=self.parent.FG_COLOR
        )
        self.label.pack(side=tk.LEFT)
        
        # Entry
        self.entry = tk.Entry(
            self.frame, 
            width=10, 
            font=self.parent.SUBHEADING_FONT,
            justify='center'
        )
        self.entry.pack(side=tk.LEFT, padx=(10, 5))
        
        # Update
        self.update_btn = tk.Button(
            self.frame, 
            text="Update", 
            command=self.update_spring_constant,
            font=self.parent.SUBHEADING_FONT,
            bg=self.parent.ICON_BUTTON_BG,
            fg=self.parent.FG_COLOR,
            relief=tk.RAISED,
            bd=1
        )
        self.update_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset
        self.reset_btn = tk.Button(
            self.frame, 
            text="Reset", 
            command=self.reset_spring_constant,
            font=self.parent.SUBHEADING_FONT,
            bg=self.parent.ICON_BUTTON_BG,
            fg=self.parent.FG_COLOR,
            relief=tk.RAISED,
            bd=1
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Readout
        self.value_label = tk.Label(
            self.frame, 
            text="", 
            font=self.parent.SUBHEADING_FONT,
            bg=self.parent.BG_COLOR, 
            fg=self.parent.FG_COLOR
        )
        self.value_label.pack(side=tk.LEFT, padx=(20, 0))
        
        Logger.log("Spring constant widget created successfully")
    
    def update_spring_constant(self):
        """Set spring constant from entry and refresh UI."""
        try:
            new_value = float(self.entry.get())
            self.controller.set_spring_constant(new_value)
            self.refresh_display()
            Logger.log(f"Spring constant updated to: {new_value}")
            messagebox.showinfo("Success", f"Spring constant updated to {new_value}\nNetwork has been relaxed with new physics parameters.")
            
        except ValueError:
            Logger.log("Invalid spring constant value entered", Logger.LogPriority.ERROR)
            messagebox.showerror("Error", "Spring constant must be a valid number")
        except Exception as ex:
            Logger.log(f"Error updating spring constant: {ex}", Logger.LogPriority.ERROR)
            messagebox.showerror("Error", f"Failed to update spring constant: {ex}")
    
    def reset_spring_constant(self):
        """Reset to original value from input metadata."""
        try:
            original = self.controller.get_original_spring_constant()
            if original is not None:
                self.controller.reset_spring_constant()
                self.entry.delete(0, tk.END)
                self.entry.insert(0, str(original))
                self.refresh_display()
                Logger.log(f"Spring constant reset to original value: {original}")
                messagebox.showinfo("Success", f"Spring constant reset to original value: {original}")
            else:
                Logger.log("No original spring constant available", Logger.LogPriority.ERROR)
                messagebox.showerror("Error", "No original spring constant found")
                
        except Exception as ex:
            Logger.log(f"Error resetting spring constant: {ex}", Logger.LogPriority.ERROR)
            messagebox.showerror("Error", f"Failed to reset spring constant: {ex}")
    
    def refresh_display(self):
        """Update readout with current and original values."""
        try:
            current = self.controller.get_spring_constant()
            original = self.controller.get_original_spring_constant()
            
            if current is not None and original is not None:
                self.value_label.config(text=f"Current: {current} | Original: {original}")
            else:
                self.value_label.config(text="No network loaded")
                
        except Exception as ex:
            Logger.log(f"Error refreshing display: {ex}", Logger.LogPriority.ERROR)
            self.value_label.config(text="Error loading values")
    
    def load_initial_value(self):
        """Populate entry with current value on load."""
        try:
            current = self.controller.get_spring_constant()
            if current is not None:
                self.entry.delete(0, tk.END)
                self.entry.insert(0, str(current))
                self.refresh_display()
                Logger.log(f"Initial spring constant loaded: {current}")
            else:
                Logger.log("No spring constant available to load")
                self.value_label.config(text="No network loaded")
                
        except Exception as ex:
            Logger.log(f"Error loading initial value: {ex}", Logger.LogPriority.ERROR)
            self.value_label.config(text="Error loading value")
    
    def clear_display(self):
        """Clear entry/readout when no network is loaded."""
        self.entry.delete(0, tk.END)
        self.value_label.config(text="No network loaded")
        Logger.log("Spring constant widget display cleared")
