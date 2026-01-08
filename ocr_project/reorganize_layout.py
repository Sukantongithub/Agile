"""
Script to reorganize the index.html layout - Version 2
Uses HTML entity-aware processing
"""

# Read the file
with open('templates/index.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# We'll build the new content line by line
new_lines = []
i = 0

while i < len(lines):
    line = lines[i]
    
    # Replacement 1: After opening preview div, add grid wrappers
    if '<div class="preview" id="previewSection">' in line and i + 1 < len(lines):
        new_lines.append(line)
        i += 1
        # Next line should be the comment
        if '<!-- Document Preview Controls -->' in lines[i]:
            new_lines.append('                <!-- Grid Layout: Left = Preview, Right = Detection/Fields -->\r\n')
            new_lines.append('                <div class="preview-grid">\r\n')
            new_lines.append('                    <!-- LEFT SIDE: Document Preview -->\r\n')
            new_lines.append('                    <div class="preview-left">\r\n')
            new_lines.append('                        <!-- Document Preview Controls -->\r\n')
            i += 1  # Skip the original comment
            continue
    
    # Replacement 2: After image container closing, close left and open right
    elif '</div>' in line and i + 2 < len(lines) and '<div class="document-type-banner"' in lines[i + 2]:
        new_lines.append(line)
        new_lines.append(lines[i + 1])  # blank line
        new_lines.append('                    </div>\r\n')
        new_lines.append('                    <!-- End preview-left -->\r\n')
        new_lines.append('\r\n')
        new_lines.append('                    <!-- RIGHT SIDE: Document Type + Fields -->\r\n')
        new_lines.append('                    <div class="preview-right">\r\n')
        new_lines.append('                        ')
        i += 2
        continue
    
    # Replacement 3: Before section-header for Detected Fields, add fields-container
    elif '<div class="section-header">' in line and i + 1 < len(lines) and '<h3>Detected Fields</h3>' in lines[i + 1]:
        new_lines.append('                        <!-- Fields Section -->\r\n')
        new_lines.append('                        <div class="fields-container">\r\n')
        new_lines.append('                            <div class="section-header">\r\n')
        i += 1
        new_lines.append('                                <h3>Detected Fields</h3>\r\n')
        i += 1
        continue
    
    # Replacement 4: Fix span badge indentation
    elif '<span class="badge">' in line and 'fieldCount' in line:
        new_lines.append('                                <span class="badge"><span id="fieldCount">0</span> Selected</span>\r\n')
        i += 1
        # Skip closing div
        if i < len(lines) and '</div>' in lines[i]:
            new_lines.append('                            </div>\r\n')
            i += 1
        continue
    
    # Replacement 5: Fix control-panel buttons
    elif '<div class="control-panel">' in line:
        new_lines.append('                            <div class="control-panel">\r\n')
        i += 1
        # Process buttons with better indentation
        while i < len(lines) and '</div>' not in lines[i]:
            btn_line = lines[i]
            if '<button' in btn_line:
                # Fix multi-line buttons
                fixed_line = btn_line.strip()
                if i + 1 < len(lines) and '</button>' not in fixed_line:
                    # Multi-line button, combine
                    fixed_line = fixed_line.replace('\r\n', '') + ' ' + lines[i + 1].strip()
                    i += 1
                new_lines.append('                                ' + fixed_line + '\r\n')
            i += 1
        new_lines.append('                            </div>\r\n')
        i += 1
        continue
    
    # Replacement 6: After fieldsList closing, close containers and start options section  
    elif '<div id="fieldsList"></div>' in line:
        new_lines.append('                            <div id="fieldsList"></div>\r\n')
        i += 1
        # Skip blank line
        if i < len(lines) and lines[i].strip() == '':
            i += 1
        # Check if next is redaction-options
        if i < len(lines) and '<div class="redaction-options">' in lines[i]:
            new_lines.append('                        </div>\r\n')
            new_lines.append('                        <!-- End fields-container -->\r\n')
            new_lines.append('                    </div>\r\n')
            new_lines.append('                    <!-- End preview-right -->\r\n')
            new_lines.append('                </div>\r\n')
            new_lines.append('                <!-- End preview-grid -->\r\n')
            new_lines.append('\r\n')
            new_lines.append('                <!-- BOTTOM SECTION: Templates + Redaction Options -->\r\n')
            new_lines.append('                <div class="options-section">\r\n')
            new_lines.append('                    <div class="redaction-options">\r\n')
            i += 1
        continue
    
    # Replacement 7: Before action-buttons, close options-section
    elif '<div class="action-buttons">' in line:
        # Check if previous line was closing div
        if len(new_lines) > 0 and '</div>' in new_lines[-1]:
            # Remove last closing div line
            new_lines.pop()
            # Check for blank line before
            if len(new_lines) > 0 and new_lines[-1].strip() == '':
                new_lines.pop()
        new_lines.append('                    </div>\r\n')
        new_lines.append('                </div>\r\n')
        new_lines.append('                <!-- End options-section -->\r\n')
        new_lines.append('\r\n')
        new_lines.append(line)
        i += 1
        continue
    
    else:
        new_lines.append(line)
        i += 1

# Write the modified content
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✅ Layout reorganization complete!")
print(f"Processed {len(lines)} lines, output {len(new_lines)} lines")
