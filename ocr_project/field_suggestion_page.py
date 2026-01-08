"""
Enhanced Field Suggestion Selection System
Adds additional interactive features for field selection and management
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import json

class FieldSelectionManager:
    """Manages field selection state and operations."""
    
    def __init__(self):
        if 'selected_fields' not in st.session_state:
            st.session_state.selected_fields = {}
        if 'field_suggestions' not in st.session_state:
            st.session_state.field_suggestions = []
    
    def render_field_suggestions_with_selection(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Render field suggestions with enhanced selection capabilities."""
        
        if not suggestions:
            st.info("🤖 No field suggestions available")
            return {}
        
        st.markdown("### 🎯 AI-Detected Fields")
        st.markdown("Select the fields you want to extract:")
        
        # Store suggestions in session state
        st.session_state.field_suggestions = suggestions
        
        # Selection controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            if st.button("✅ Select All", key="select_all"):
                self._select_all_fields()
                st.rerun()
        
        with col2:
            if st.button("❌ Clear All", key="clear_all"):
                self._clear_all_fields()
                st.rerun()
        
        with col3:
            confidence_filter = st.selectbox(
                "Filter by Confidence",
                options=["All", "High (>80%)", "Medium (>60%)", "Low (>40%)"],
                key="confidence_filter"
            )
        
        with col4:
            category_filter = st.selectbox(
                "Filter by Category", 
                options=["All"] + list(set([s.get('field_category', 'unknown') for s in suggestions])),
                key="category_filter"
            )
        
        # Filter suggestions based on selection
        filtered_suggestions = self._filter_suggestions(suggestions, confidence_filter, category_filter)
        
        # Display filtered suggestions with selection
        selected_fields = {}
        
        for i, suggestion in enumerate(filtered_suggestions):
            field_key = f"field_{i}_{suggestion['field_name']}"
            
            # Create expandable field item
            with st.expander(
                f"{'✅' if field_key in st.session_state.selected_fields else '⬜'} "
                f"{suggestion['field_name']} - {suggestion.get('value', 'N/A')} "
                f"({suggestion.get('confidence', 0)*100:.1f}%)",
                expanded=field_key in st.session_state.selected_fields
            ):
                # Selection checkbox
                is_selected = st.checkbox(
                    f"Select {suggestion['field_name']}", 
                    value=field_key in st.session_state.selected_fields,
                    key=f"checkbox_{field_key}"
                )
                
                if is_selected:
                    st.session_state.selected_fields[field_key] = suggestion
                    selected_fields[field_key] = suggestion
                elif field_key in st.session_state.selected_fields:
                    del st.session_state.selected_fields[field_key]
                
                # Field details
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Field Type:**", suggestion.get('field_type', 'Unknown'))
                    st.write("**Category:**", suggestion.get('field_category', 'Unknown').replace('_', ' ').title())
                    st.write("**Confidence:**", f"{suggestion.get('confidence', 0)*100:.1f}%")
                
                with col2:
                    st.write("**Extracted Value:**")
                    st.code(suggestion.get('value', 'No value'))
                    
                    if st.button(f"📋 Copy Value", key=f"copy_{field_key}"):
                        st.write("✅ Value copied to clipboard!")
                
                # Reasoning
                if 'reasoning' in suggestion:
                    st.write("**AI Reasoning:**", suggestion['reasoning'])
        
        # Show selection summary
        self._render_selection_summary()
        
        return st.session_state.selected_fields
    
    def _filter_suggestions(self, suggestions: List[Dict], confidence_filter: str, category_filter: str) -> List[Dict]:
        """Filter suggestions based on confidence and category."""
        filtered = suggestions.copy()
        
        # Filter by confidence
        if confidence_filter == "High (>80%)":
            filtered = [s for s in filtered if s.get('confidence', 0) > 0.8]
        elif confidence_filter == "Medium (>60%)":
            filtered = [s for s in filtered if s.get('confidence', 0) > 0.6]
        elif confidence_filter == "Low (>40%)":
            filtered = [s for s in filtered if s.get('confidence', 0) > 0.4]
        
        # Filter by category
        if category_filter != "All":
            filtered = [s for s in filtered if s.get('field_category') == category_filter]
        
        return filtered
    
    def _select_all_fields(self):
        """Select all available fields."""
        for i, suggestion in enumerate(st.session_state.field_suggestions):
            field_key = f"field_{i}_{suggestion['field_name']}"
            st.session_state.selected_fields[field_key] = suggestion
    
    def _clear_all_fields(self):
        """Clear all selected fields."""
        st.session_state.selected_fields = {}
    
    def _render_selection_summary(self):
        """Render summary of selected fields."""
        if not st.session_state.selected_fields:
            return
        
        st.markdown("---")
        st.markdown(f"### 📊 Selected Fields ({len(st.session_state.selected_fields)})")
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            if st.button("📄 Generate Form", key="generate_form"):
                self._generate_form_from_selections()
        
        with col2:
            if st.button("💾 Export JSON", key="export_json"):
                self._export_selected_as_json()
        
        with col3:
            if st.button("📋 Copy All", key="copy_all"):
                self._copy_selected_fields()
        
        # Display selected fields in a nice format
        for field_key, field_data in st.session_state.selected_fields.items():
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{field_data['field_name']}**")
                    st.caption(field_data.get('field_category', 'Unknown').replace('_', ' ').title())
                
                with col2:
                    st.code(field_data.get('value', 'No value'))
                
                with col3:
                    confidence = field_data.get('confidence', 0) * 100
                    if confidence > 80:
                        st.success(f"{confidence:.1f}%")
                    elif confidence > 60:
                        st.warning(f"{confidence:.1f}%")
                    else:
                        st.error(f"{confidence:.1f}%")
    
    def _generate_form_from_selections(self):
        """Generate an editable form from selected fields."""
        st.markdown("### 📝 Editable Form")
        
        edited_values = {}
        for field_key, field_data in st.session_state.selected_fields.items():
            field_name = field_data['field_name']
            current_value = field_data.get('value', '')
            
            # Create input field based on field type
            field_type = field_data.get('field_type', 'text')
            
            if field_type in ['email', 'phone', 'name']:
                edited_values[field_name] = st.text_input(
                    f"{field_name.replace('_', ' ').title()}:",
                    value=current_value,
                    key=f"edit_{field_key}"
                )
            elif field_type == 'date':
                try:
                    import datetime
                    if current_value:
                        # Try to parse the date
                        date_value = datetime.datetime.strptime(current_value, "%Y-%m-%d").date()
                    else:
                        date_value = None
                    edited_values[field_name] = st.date_input(
                        f"{field_name.replace('_', ' ').title()}:",
                        value=date_value,
                        key=f"edit_{field_key}"
                    )
                except:
                    edited_values[field_name] = st.text_input(
                        f"{field_name.replace('_', ' ').title()}: (Date)",
                        value=current_value,
                        key=f"edit_{field_key}"
                    )
            elif field_type == 'number':
                try:
                    num_value = float(current_value) if current_value else 0.0
                    edited_values[field_name] = st.number_input(
                        f"{field_name.replace('_', ' ').title()}:",
                        value=num_value,
                        key=f"edit_{field_key}"
                    )
                except:
                    edited_values[field_name] = st.text_input(
                        f"{field_name.replace('_', ' ').title()}: (Number)",
                        value=current_value,
                        key=f"edit_{field_key}"
                    )
            else:
                edited_values[field_name] = st.text_area(
                    f"{field_name.replace('_', ' ').title()}:",
                    value=current_value,
                    key=f"edit_{field_key}",
                    height=50
                )
        
        if st.button("💾 Save Edited Form", key="save_form"):
            st.session_state.edited_form_data = edited_values
            st.success("✅ Form saved successfully!")
            
            # Show the saved form data
            st.json(edited_values)
    
    def _export_selected_as_json(self):
        """Export selected fields as JSON."""
        export_data = []
        for field_key, field_data in st.session_state.selected_fields.items():
            export_data.append({
                'field_name': field_data['field_name'],
                'field_type': field_data.get('field_type', 'unknown'),
                'field_category': field_data.get('field_category', 'unknown'),
                'value': field_data.get('value', ''),
                'confidence': field_data.get('confidence', 0.0),
                'reasoning': field_data.get('reasoning', 'AI-detected field')
            })
        
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"selected_fields_{len(export_data)}.json",
            mime="application/json",
            key="download_json"
        )
        
        st.success(f"✅ Ready to download {len(export_data)} selected fields as JSON!")
    
    def _copy_selected_fields(self):
        """Copy selected fields to clipboard (simplified for demo)."""
        copy_text = ""
        for field_key, field_data in st.session_state.selected_fields.items():
            copy_text += f"{field_data['field_name']}: {field_data.get('value', '')}\n"
        
        st.code(copy_text)
        st.info("📋 Copy the text above to your clipboard")

def demo_field_selection():
    """Demo function to show the enhanced field selection system."""
    
    st.title("🎯 Enhanced Field Selection Demo")
    st.markdown("This demo shows the enhanced field suggestion selection system")
    
    # Mock field suggestions for demo
    mock_suggestions = [
        {
            "field_name": "customer_name",
            "field_type": "name",
            "field_category": "personal_info",
            "value": "John Smith",
            "confidence": 0.95,
            "reasoning": "Found name pattern in header section"
        },
        {
            "field_name": "email_address", 
            "field_type": "email",
            "field_category": "contact_info",
            "value": "john.smith@email.com",
            "confidence": 0.88,
            "reasoning": "Email format detected with @ symbol"
        },
        {
            "field_name": "phone_number",
            "field_type": "phone", 
            "field_category": "contact_info",
            "value": "+1 234-567-8900",
            "confidence": 0.76,
            "reasoning": "Phone number pattern with country code"
        },
        {
            "field_name": "total_amount",
            "field_type": "currency",
            "field_category": "financial_info", 
            "value": "$1,234.56",
            "confidence": 0.92,
            "reasoning": "Currency symbol and amount pattern found"
        },
        {
            "field_name": "date_of_birth",
            "field_type": "date",
            "field_category": "personal_info",
            "value": "1985-03-15",
            "confidence": 0.67,
            "reasoning": "Date pattern in personal information section"
        }
    ]
    
    # Initialize field selection manager
    manager = FieldSelectionManager()
    
    # Render the enhanced field selection interface
    selected_fields = manager.render_field_suggestions_with_selection(mock_suggestions)
    
    # Show results
    if selected_fields:
        st.markdown("---")
        st.markdown("### 🎉 Selection Results")
        st.write(f"You have selected **{len(selected_fields)}** fields")

if __name__ == "__main__":
    demo_field_selection()
