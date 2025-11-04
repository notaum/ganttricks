import sys
import csv
import re
import colorsys
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
import argparse
import os


def parse_date(date_str):
    date_str = date_str.strip().rstrip('.')
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_str)
    if not match:
        raise ValueError(f"Cannot parse date: {date_str}")
    d, m, y = map(int, match.groups())
    try:
        return date(y, m, d)
    except ValueError as e:
        raise ValueError(f"Invalid date: {date_str}") from e

def ordinal(n):
    if 11 <= (n % 100) <= 13:
        s = 'th'
    else:
        s = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + s

def create_shade(hex_color, shade_factor):
    """Create a shade by adjusting lightness of a hex color"""
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Adjust lightness to create shade (darker)
    l_adjusted = l * (1 - shade_factor * 0.6)
    l_adjusted = max(0.3, l_adjusted)  # Prevent too dark
    r, g, b = colorsys.hls_to_rgb(h, l_adjusted, s)
    r = int(r * 255 + 0.5)
    g = int(g * 255 + 0.5)
    b = int(b * 255 + 0.5)
    return f'#{r:02x}{g:02x}{b:02x}'

def get_week_start_date(dt):
    """Get Monday of the week for a given date"""
    # Monday is weekday 0, Sunday is 6
    return dt - timedelta(days=dt.weekday())

def is_first_week_of_month(dt):
    """Check if a date is in the first week of its month"""
    # Get the first day of the month
    first_day = dt.replace(day=1)
    # Get the Monday of the week containing the first day
    first_week_monday = get_week_start_date(first_day)
    # Check if our date is in that same week
    return get_week_start_date(dt) == first_week_monday

def resolve_overlaps(df):
    """
    Resolve overlapping tasks within the same project by creating sub-groups.
    Returns a modified DataFrame with a new 'SubGroup' column.
    """
    # Make a copy to avoid modifying the original
    df_modified = df.copy()
    
    # Add subgroup column - will be filled for overlapping tasks
    df_modified['SubGroup'] = 0
    
    # Process each project separately
    for project in df_modified['Project'].unique():
        project_tasks = df_modified[df_modified['Project'] == project].copy()
        
        if len(project_tasks) <= 1:
            # No overlaps possible with only one task
            continue
            
        # Sort tasks by start date
        project_tasks = project_tasks.sort_values('Start')
        
        # Initialize subgroups
        subgroups = [[]]  # List of subgroups, each subgroup is a list of task indices
        
        for idx, task in project_tasks.iterrows():
            placed = False
            
            # Try to place in existing subgroups
            for subgroup_idx, subgroup in enumerate(subgroups):
                can_place = True
                
                # Check if this task overlaps with any task in the current subgroup
                for existing_task_idx in subgroup:
                    existing_task = project_tasks.loc[existing_task_idx]
                    
                    # Check for overlap: new task starts before existing task ends AND
                    # new task ends after existing task starts
                    if (task['Start'] < existing_task['Finish'] and 
                        task['Finish'] > existing_task['Start']):
                        can_place = False
                        break
                
                if can_place:
                    # Place task in this subgroup
                    subgroup.append(idx)
                    df_modified.at[idx, 'SubGroup'] = subgroup_idx
                    placed = True
                    break
            
            if not placed:
                # Create new subgroup
                subgroups.append([idx])
                df_modified.at[idx, 'SubGroup'] = len(subgroups) - 1
    
    return df_modified

def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive Gantt charts from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s projects.csv                    # Basic chart with automatic overlap handling
  %(prog)s projects.csv --no-overlap       # Disable overlap handling (old behavior)
  %(prog)s projects.csv --project-sort    # Colors by project with shading
  %(prog)s projects.csv --dark            # Dark theme
  %(prog)s projects.csv --name "Q4 Timeline" --graph weekly
  %(prog)s projects.csv --height 30 --opacity 80 --clean

CSV Format:
  Project A, Start, End, Project B, Start, End
  Task 1, 1/1/2023, 15/1/2023, Task X, 10/1/2023, 20/1/2023
  Task 2, 16/1/2023, 30/1/2023, Task Y, 25/1/2023, 5/2/2023
        """
    )
    
    # Required argument
    parser.add_argument('input_file', help='Input CSV file with project data')
    
    # Chart customization group
    chart_group = parser.add_argument_group('chart customization')
    chart_group.add_argument('--name', type=str, 
                           help='Custom chart title (default: filename)')
    chart_group.add_argument('--dark', action='store_true', 
                           help='Use dark theme')
    chart_group.add_argument('--height', type=int, default=0,
                           help='Height adjustment (%%): +40 for 40%% taller, -30 for shorter')
    chart_group.add_argument('--opacity', type=int, default=100,
                           help='Bar opacity percentage (0-100, default: 100)')
    chart_group.add_argument('--rounding', type=int, default=5,
                           help='Bar corner radius in pixels (default: 5)')
    chart_group.add_argument('--project-sort', action='store_true',
                           help='Color by project (default: color by task)')
    
    # Layout group
    layout_group = parser.add_argument_group('layout options')
    layout_group.add_argument('--graph', choices=['daily', 'weekly', 'weekly-name', 'monthly'], 
                            default='monthly',
                            help='Grid line frequency (default: monthly)')
    layout_group.add_argument('--clean', action='store_true', 
                            help='Hide x-axis labels for cleaner look')
    
    # Changed overlap handling argument - now default is ON, use --no-overlap to disable
    layout_group.add_argument('--no-overlap', action='store_true',
                            help='Disable automatic handling of overlapping tasks within projects')
    
    args = parser.parse_args()
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]

    # Convert opacity percentage to float (0.0-1.0)
    opacity_float = args.opacity / 100.0
    
    # Validate arguments
    if args.opacity < 0 or args.opacity > 100:
        print("Error: Opacity must be between 0 and 100")
        sys.exit(1)
    
    if args.rounding < 0:
        print("Error: Rounding must be a non-negative integer")
        sys.exit(1)

    # Determine chart title
    if args.name:
        chart_title = args.name
    else:
        # Use input file name without extension
        chart_title = f"{base_name} Timeline"

    try:
        with open(args.input_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("Empty file")
        header = [col.strip() for col in rows[0]]
        data_rows = [[cell.strip() for cell in row] for row in rows[1:]]

        # Find projects
        projects = []
        for col in header:
            if col not in ['Start', 'End']:
                projects.append(col)
        num_projs = len(projects)
        if len(header) != num_projs * 3:
            print("Header format mismatch. Expected format: ProjectName,Start,End,...")
            sys.exit(1)

        # Now parse data
        parsed_data = []
        for row in data_rows:
            for k in range(num_projs):
                task_col = k * 3
                if task_col >= len(row):
                    break
                task = row[task_col]
                if not task:
                    continue
                start_str = row[task_col + 1] if task_col + 1 < len(row) else ''
                end_str = row[task_col + 2] if task_col + 2 < len(row) else ''
                if not start_str or not end_str:
                    continue
                try:
                    start = parse_date(start_str)
                    end = parse_date(end_str)
                    parsed_data.append((projects[k], task, start, end))
                except ValueError as e:
                    print(f"Skipping invalid date in row {row}: {e}")
                    continue

        # Build DataFrame
        data = []
        for proj, task, st, en in parsed_data:
            data.append({'Project': proj, 'Task': task, 'Start': st, 'Finish': en})

        if not data:
            print("No valid tasks found.")
            sys.exit(1)

        df = pd.DataFrame(data)

        # Handle overlapping tasks (now the DEFAULT behavior)
        # FIXED: Changed args.nooverlap to args.no_overlap (with underscore)
        if not args.no_overlap:  
            print("Resolving overlapping tasks...")
            df = resolve_overlaps(df)
            
            # Create a new y-axis column that combines project and subgroup for non-overlapping display
            df['DisplayProject'] = df['Project'] + df['SubGroup'].apply(lambda x: f" ({chr(65 + x)})" if x > 0 else "")
        else:
            # Keep original project name for display when --no-overlap is used
            df['DisplayProject'] = df['Project']

        # Prepare colors based on the coloring mode
        if args.project_sort:
            # OLD BEHAVIOR: Color by project with shading for tasks within each project
            unique_projs = sorted(df['Project'].unique())
            n_projs = len(unique_projs)
            color_sets = px.colors.qualitative.Plotly
            base_colors_list = (color_sets * ((n_projs // len(color_sets)) + 1))[:n_projs]
            base_colors = dict(zip(unique_projs, base_colors_list))

            # Create a unique identifier for each task by combining project and task
            df['Color_Group'] = df['Project'] + ' - ' + df['Task']

            # Create color mapping for each unique task with shading within projects
            color_discrete_map = {}
            
            # Group by project and sort tasks by start date within each project
            for project in unique_projs:
                project_tasks = df[df['Project'] == project].sort_values('Start')
                tasks = project_tasks['Color_Group'].tolist()
                n_tasks = len(tasks)
                
                base_color = base_colors[project]
                
                # Assign shades to tasks based on their order
                for i, task_unique in enumerate(tasks):
                    # Calculate shade factor (0 for first task, 1 for last task)
                    shade_factor = i / max(n_tasks - 1, 1) if n_tasks > 1 else 0
                    color = create_shade(base_color, shade_factor)
                    color_discrete_map[task_unique] = color

            # Use the project-task combination for coloring
            color_column = 'Color_Group'
            
        else:
            # NEW DEFAULT BEHAVIOR: Color by task (same task name = same color across projects)
            unique_tasks = sorted(df['Task'].unique())
            n_tasks = len(unique_tasks)
            color_sets = px.colors.qualitative.Plotly
            base_colors_list = (color_sets * ((n_tasks // len(color_sets)) + 1))[:n_tasks]
            
            # Create direct mapping from task name to color
            color_discrete_map = dict(zip(unique_tasks, base_colors_list))
            
            # Use just the task name for coloring
            color_column = 'Task'

        # Sort for ordering - always sort by project and start date for consistent display
        df_sorted = df.sort_values(['Project', 'Start'])

        # Create figure - use the appropriate color column based on mode
        # For overlap handling, use DisplayProject which includes subgroup indicators
        fig = px.timeline(
            df_sorted,
            x_start='Start',
            x_end='Finish',
            y='DisplayProject',  # Use the display column that may include subgroup info
            color=color_column,
            color_discrete_map=color_discrete_map
        )

        # Update layout for good looks
        template = 'plotly_dark' if args.dark else 'plotly_white'
        output_suffix = '_dark' if args.dark else ''
        grid_color = "#4D4D4D" if args.dark else "#D2D5D9"
        daily_line_color = "#4D4D4D" if args.dark else "#C7C7C7"
        text_color = "white" if args.dark else "black"
        paper_bgcolor = "#1e1e1e" if args.dark else "white"
        plot_bgcolor = "#2d2d2d" if args.dark else "white"

        # Calculate height using percentage adjustment (like original code)
        # Adjust height based on number of unique display projects (which may include subgroups)
        unique_display_projects = len(df['DisplayProject'].unique())
        default_height = max(400, unique_display_projects * 100)
        height_multiplier = 1 + args.height / 100
        height = int(default_height * height_multiplier)

        # Ensure minimum height
        height = max(200, height)

        # Extend x-axis by 1 month on both sides
        start_date = df['Start'].min()
        end_date = df['Finish'].max()
        extended_start = start_date - timedelta(days=30)
        extended_end = end_date + timedelta(days=30)

        # Add vertical lines based on the graph frequency
        min_date = extended_start
        max_date = extended_end
        shapes = []
        
        if args.graph == 'daily':
            # Daily lines
            current_date = min_date
            while current_date <= max_date:
                shapes.append(dict(
                    type="line",
                    x0=current_date,
                    x1=current_date,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(
                        color=daily_line_color,
                        width=0.5,
                    ),
                    layer="below"
                ))
                current_date += timedelta(days=1)
                
        elif args.graph in ['weekly', 'weekly-name']:
            # Weekly lines - every Monday
            current_date = get_week_start_date(min_date)
            
            while current_date <= max_date:
                shapes.append(dict(
                    type="line",
                    x0=current_date,
                    x1=current_date,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(
                        color=daily_line_color,
                        width=0.8,  # Slightly thicker than daily but thinner than monthly
                    ),
                    layer="below"
                ))
                current_date += timedelta(days=7)
                
        elif args.graph == 'monthly':
            # Monthly lines - 1st of each month
            current_date = min_date.replace(day=1)
            
            while current_date <= max_date:
                shapes.append(dict(
                    type="line",
                    x0=current_date,
                    x1=current_date,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(
                        color=daily_line_color,
                        width=1.0,  # Thicker for monthly lines
                    ),
                    layer="below"
                ))
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1, day=1)

        # Update x-axis - show or hide labels based on clean mode
        if args.clean:
            # In clean mode, show no labels but keep grid lines and vertical lines
            fig.update_xaxes(
                showticklabels=False,  # Hide all tick labels
                tickangle=-45,
                gridcolor=grid_color,
                gridwidth=0.5
            )
        else:
            # Generate custom labels based on graph frequency for non-clean mode
            tickvals = []
            ticktext = []
            
            if args.graph == 'monthly':
                # Monthly labels: "Month" (e.g., "Jan", "Feb", etc.)
                current_date = extended_start.replace(day=1)
                
                while current_date <= extended_end:
                    tickvals.append(current_date)
                    month_abbr = current_date.strftime('%b')
                    ticktext.append(month_abbr)
                    # Move to next month
                    if current_date.month == 12:
                        current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
                    else:
                        current_date = current_date.replace(month=current_date.month + 1, day=1)
                        
            elif args.graph == 'weekly':
                # Weekly labels: Show Monday dates in "DD/MM" format (day/month)
                current_date = get_week_start_date(extended_start)
                
                while current_date <= extended_end:
                    tickvals.append(current_date)
                    # Format as "DD/MM" for weekly labels (day/month)
                    ticktext.append(current_date.strftime('%d/%m'))
                    current_date += timedelta(days=7)
                    
            elif args.graph == 'weekly-name':
                # Weekly labels with month names for first week of each month
                current_date = get_week_start_date(extended_start)
                last_month = None
                
                while current_date <= extended_end:
                    tickvals.append(current_date)
                    
                    # Add month name for first week of each month
                    if current_date.month != last_month:
                        month_abbr = current_date.strftime('%b')
                        label = f"{current_date.strftime('%d/%m')} {month_abbr}"
                        last_month = current_date.month
                    else:
                        label = current_date.strftime('%d/%m')
                    
                    ticktext.append(label)
                    current_date += timedelta(days=7)
                    
            elif args.graph == 'daily':
                # Daily labels: 1st of every month + every alternate day (odd days)
                current_date = extended_start
                
                while current_date <= extended_end:
                    # Always label the 1st of each month
                    if current_date.day == 1:
                        tickvals.append(current_date)
                        month_abbr = current_date.strftime('%b')
                        ticktext.append(f"1 {month_abbr}")
                    # Label every alternate day (odd days) except the 1st
                    elif current_date.day % 2 == 1:  # Odd day
                        tickvals.append(current_date)
                        ticktext.append(str(current_date.day))
                    current_date += timedelta(days=1)
            
            # Normal mode with labels
            fig.update_xaxes(
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=-45,
                gridcolor=grid_color,
                gridwidth=0.5
            )

        # Order y-axis by display project (which maintains project grouping)
        unique_display_projects = sorted(df['DisplayProject'].unique())
        fig.update_yaxes(categoryorder='array', categoryarray=unique_display_projects)

        # Style traces - use the new opacity and rounding arguments
        fig.update_traces(
            marker=dict(
                line_width=1,
                opacity=opacity_float,  # Use the converted opacity value
                cornerradius=args.rounding
            )
        )

        # Update layout with shapes and adjust margins to prevent bar cutoff
        fig.update_layout(
            title=chart_title,  # Use the dynamically determined chart title
            xaxis_title='Date',
            yaxis_title='Projects',
            height=height,
            showlegend=True,
            xaxis=dict(
                rangeslider=dict(visible=True),
                range=[extended_start, extended_end]
            ),
            template=template,
            font=dict(color=text_color),
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            shapes=shapes,
            margin=dict(l=50, r=50, t=50, b=50)  # Fixed margins to prevent cutoff
        )

        fig.show()

        # Export to HTML
        output_file = f'{base_name}{output_suffix}.html'
        fig.write_html(output_file, include_plotlyjs='cdn')
        print(f"Chart exported to {output_file}")
        
    except FileNotFoundError:
        print(f"File not found: {args.input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
