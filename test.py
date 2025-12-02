import streamlit as st
import pandas as pd
import time
from datetime import datetime
import json

# Configure page
st.set_page_config(
    page_title="User Testing - Multimodal Viz",
    page_icon="🧪",
    layout="wide"
)

# Initialize session state
if 'participant_id' not in st.session_state:
    st.session_state.participant_id = None
if 'current_task' not in st.session_state:
    st.session_state.current_task = 0
if 'task_start_time' not in st.session_state:
    st.session_state.task_start_time = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'system_type' not in st.session_state:
    st.session_state.system_type = None

# Define testing tasks
TASKS = [
    {
        'id': 1,
        'description': 'Switch from Bar Chart to Line Chart',
        'expected_action': 'Change visualization type',
        'success_criteria': 'Chart type changed successfully'
    },
    {
        'id': 2,
        'description': 'Apply a filter to show only high values (above median)',
        'expected_action': 'Filter data',
        'success_criteria': 'Data filtered correctly'
    },
    {
        'id': 3,
        'description': 'Compare two different chart types side by side',
        'expected_action': 'Enable comparison mode and select charts',
        'success_criteria': 'Multiple charts displayed'
    },
    {
        'id': 4,
        'description': 'Switch to Pie Chart and then to Scatter Plot',
        'expected_action': 'Navigate through chart types',
        'success_criteria': 'Successfully navigated to both charts'
    },
    {
        'id': 5,
        'description': 'Remove all filters and view complete dataset',
        'expected_action': 'Clear filters',
        'success_criteria': 'All data displayed'
    }
]

def save_test_results(results):
    """Save test results to a JSON file"""
    filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    return filename

def calculate_metrics(results):
    """Calculate performance metrics from test results"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    metrics = {
        'total_participants': len(df['participant_id'].unique()),
        'avg_completion_time': df['completion_time'].mean(),
        'task_completion_rate': (df['task_completed'].sum() / len(df)) * 100,
        'avg_difficulty': df['difficulty_rating'].mean(),
        'avg_satisfaction': df['satisfaction_rating'].mean(),
        'avg_ease_of_use': df['ease_of_use_rating'].mean(),
        'total_errors': df['errors_count'].sum(),
        'avg_errors_per_task': df['errors_count'].mean()
    }
    
    return metrics

# Main Testing Interface
def main():
    st.title("🧪 User Testing Framework")
    st.markdown("### Multimodal vs Traditional Data Visualization")
    st.markdown("---")
    
    # Participant Registration
    if st.session_state.participant_id is None:
        st.subheader("📋 Participant Registration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            participant_name = st.text_input("Participant Name")
            age = st.number_input("Age", min_value=18, max_value=100, value=25)
            occupation = st.text_input("Occupation/Role")
        
        with col2:
            experience_level = st.selectbox(
                "Data Analysis Experience",
                ["Beginner", "Intermediate", "Advanced", "Expert"]
            )
            viz_tools_used = st.multiselect(
                "Visualization Tools Experience",
                ["Tableau", "Power BI", "Excel", "Python", "R", "None"]
            )
        
        st.markdown("---")
        
        system_type = st.radio(
            "Select System to Test",
            ["Multimodal System (Gesture + Voice)", "Traditional System (Mouse + Keyboard)"],
            help="You will test this system first, then the other system"
        )
        
        if st.button("Start Testing", type="primary"):
            if participant_name:
                st.session_state.participant_id = f"P{len(st.session_state.test_results) + 1:03d}"
                st.session_state.system_type = system_type
                st.session_state.participant_info = {
                    'id': st.session_state.participant_id,
                    'name': participant_name,
                    'age': age,
                    'occupation': occupation,
                    'experience': experience_level,
                    'tools_used': viz_tools_used,
                    'test_date': datetime.now().isoformat()
                }
                st.rerun()
            else:
                st.error("Please enter your name to continue")
    
    else:
        # Display participant info
        st.info(f"**Participant ID:** {st.session_state.participant_id} | **Testing:** {st.session_state.system_type}")
        
        # Task interface
        if st.session_state.current_task < len(TASKS):
            current_task = TASKS[st.session_state.current_task]
            
            st.subheader(f"Task {current_task['id']} of {len(TASKS)}")
            
            st.markdown(f"""
            ### 📌 Task Description:
            **{current_task['description']}**
            
            **Expected Action:** {current_task['expected_action']}
            
            **Success Criteria:** {current_task['success_criteria']}
            """)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.session_state.task_start_time is None:
                    if st.button("▶️ Start Task", type="primary"):
                        st.session_state.task_start_time = time.time()
                        st.rerun()
                else:
                    elapsed = time.time() - st.session_state.task_start_time
                    st.metric("⏱️ Elapsed Time", f"{elapsed:.1f} seconds")
                    
                    st.markdown("---")
                    
                    # Task completion form
                    st.markdown("### Complete Task")
                    
                    task_completed = st.radio(
                        "Did you complete the task successfully?",
                        ["Yes", "No", "Partially"]
                    )
                    
                    errors_count = st.number_input(
                        "Number of errors/attempts made",
                        min_value=0,
                        value=0,
                        help="How many times did you make a mistake or retry?"
                    )
                    
                    difficulty_rating = st.slider(
                        "Task Difficulty (1=Very Easy, 5=Very Difficult)",
                        1, 5, 3
                    )
                    
                    satisfaction_rating = st.slider(
                        "Satisfaction with completing this task (1=Very Unsatisfied, 5=Very Satisfied)",
                        1, 5, 3
                    )
                    
                    ease_of_use = st.slider(
                        "Ease of Use (1=Very Difficult, 5=Very Easy)",
                        1, 5, 3
                    )
                    
                    comments = st.text_area(
                        "Additional Comments/Observations",
                        placeholder="Any challenges, thoughts, or suggestions?"
                    )
                    
                    if st.button("✅ Submit Task Results", type="primary"):
                        completion_time = time.time() - st.session_state.task_start_time
                        
                        task_result = {
                            'participant_id': st.session_state.participant_id,
                            'system_type': st.session_state.system_type,
                            'task_id': current_task['id'],
                            'task_description': current_task['description'],
                            'completion_time': round(completion_time, 2),
                            'task_completed': task_completed == "Yes",
                            'errors_count': errors_count,
                            'difficulty_rating': difficulty_rating,
                            'satisfaction_rating': satisfaction_rating,
                            'ease_of_use_rating': ease_of_use,
                            'comments': comments,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        st.session_state.test_results.append(task_result)
                        st.session_state.current_task += 1
                        st.session_state.task_start_time = None
                        st.success("Task results recorded!")
                        time.sleep(1)
                        st.rerun()
            
            with col2:
                st.markdown("### 📊 Progress")
                st.progress((st.session_state.current_task) / len(TASKS))
                st.metric("Completed", f"{st.session_state.current_task}/{len(TASKS)}")
        
        else:
            # Testing completed
            st.success("🎉 All tasks completed!")
            
            st.markdown("### 📋 Post-Test Questionnaire")
            
            overall_satisfaction = st.slider(
                "Overall Satisfaction with the system (1=Very Unsatisfied, 10=Very Satisfied)",
                1, 10, 5
            )
            
            would_recommend = st.radio(
                "Would you recommend this system to others?",
                ["Yes", "No", "Maybe"]
            )
            
            preferred_system = st.radio(
                "Which system did you prefer overall?",
                ["Multimodal (Gesture + Voice)", "Traditional (Mouse + Keyboard)", "No Preference"]
            )
            
            strengths = st.text_area(
                "What were the strengths of this system?",
                placeholder="List the positive aspects..."
            )
            
            weaknesses = st.text_area(
                "What were the weaknesses or areas for improvement?",
                placeholder="List areas that need improvement..."
            )
            
            additional_features = st.text_area(
                "What additional features would you like to see?",
                placeholder="Suggest new features..."
            )
            
            if st.button("📤 Submit Final Feedback", type="primary"):
                final_feedback = {
                    'participant_id': st.session_state.participant_id,
                    'system_type': st.session_state.system_type,
                    'overall_satisfaction': overall_satisfaction,
                    'would_recommend': would_recommend,
                    'preferred_system': preferred_system,
                    'strengths': strengths,
                    'weaknesses': weaknesses,
                    'additional_features': additional_features,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.test_results.append(final_feedback)
                
                # Save results
                filename = save_test_results(st.session_state.test_results)
                st.success(f"✅ Results saved to {filename}")
                
                # Reset for next participant
                st.session_state.participant_id = None
                st.session_state.current_task = 0
                st.session_state.task_start_time = None
                
                st.balloons()
                time.sleep(2)
                st.rerun()
    
    # Admin section - View Results
    st.markdown("---")
    with st.expander("👨‍💼 Admin: View All Results"):
        if st.session_state.test_results:
            results_df = pd.DataFrame([
                r for r in st.session_state.test_results 
                if 'task_id' in r
            ])
            
            if not results_df.empty:
                st.dataframe(results_df, use_container_width=True)
                
                # Calculate metrics
                st.subheader("📊 Performance Metrics")
                metrics = calculate_metrics(st.session_state.test_results)
                
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Task Completion Rate", f"{metrics['task_completion_rate']:.1f}%")
                        st.metric("Avg Completion Time", f"{metrics['avg_completion_time']:.1f}s")
                    
                    with col2:
                        st.metric("Avg Difficulty", f"{metrics['avg_difficulty']:.1f}/5")
                        st.metric("Avg Satisfaction", f"{metrics['avg_satisfaction']:.1f}/5")
                    
                    with col3:
                        st.metric("Avg Ease of Use", f"{metrics['avg_ease_of_use']:.1f}/5")
                        st.metric("Total Errors", f"{metrics['total_errors']:.0f}")
                    
                    with col4:
                        st.metric("Total Participants", metrics['total_participants'])
                        st.metric("Avg Errors/Task", f"{metrics['avg_errors_per_task']:.1f}")
                
                # Download results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Results as CSV",
                    csv,
                    "testing_results.csv",
                    "text/csv"
                )
        else:
            st.info("No test results yet")

if __name__ == "__main__":
    main()