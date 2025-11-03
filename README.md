# Ganttricks

The only free Multi-Project Timeline Visualizer!
(I searchead and couldnt find one)

Like a gantt chart but with multiple tasks in each bar of the timeline view. 
Useful for managing multiple projects or multiple milestones within a project.

    🎨 Interactive charts - Hover, zoom, and pan your timeline

    🌙 Dark/Light themes - Choose your preferred theme

    🎯 Smart coloring - Automatic color coordination within projects

    📱 Fully customizable - Adjust height, opacity, rounding, and more

    📊 Multiple grid styles - Daily, weekly, or monthly grid lines

    💾 Export to HTML - Share interactive charts easily

## Quick Start

Install python then run

```bash
pip install pandas plotly
```
Once pandas and plotly libraries are installed run the code with 

```bash
python ganttricks.py -h
```

## CSV Format

Your CSV **MUST** follow this structure

```csv
Project A, Start, End, Project B, Start, End
Task 1, 1/1/2023, 15/1/2023, Task X, 10/1/2023, 20/1/2023
Task 2, 16/1/2023, 30/1/2023, Task Y, 25/1/2023, 5/2/2023
```

DATES MUST BE IN DD/MM/YYYY otherwise the script will not work.

## Command Line Options

### Chart Customization

    --name - Custom chart title (default: filename)

    --dark - Use dark theme

    --height - Height adjustment percentage (e.g., 40 for 40% taller)

    --opacity - Bar opacity percentage (0-100, default: 100)

    --rounding - Bar corner radius in pixels (default: 5)

### Layout Options

    --graph - Grid line frequency: daily, weekly, weekly-name, monthly (default)

    --clean - Hide x-axis labels for cleaner look




