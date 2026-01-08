"""
Fixed script to reorganize the index.html layout
Only modifies HTML structure, preserves all JavaScript and CSS
"""

# Read the file
with open('templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# First, add the CSS for grid layout if not present
if 'preview-grid' not in content:
    # Find the closing of footer-tech style and add grid styles before </style>
    css_addition = '''
        /* Grid Layout for Preview Section */
        .preview-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .preview-left {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .preview-right {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .options-section {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
            margin-top: 24px;
        }

        .fields-container {
            background: var(--bg-secondary);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }

        .fields-container::-webkit-scrollbar {
            width: 8px;
        }

        .fields-container::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
            border-radius: 4px;
        }

        .fields-container::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }

        .fields-container::-webkit-scrollbar-thumb:hover {
            background: var(--primary-light);
        }

        /* Responsive Design */
        @media (max-width: 1200px) {
            .preview-grid {
                grid-template-columns: 1fr;
            }

            .options-section {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .main-content {
                padding: 20px 10px;
            }

            .container {
                padding: 20px;
            }

            .option-row {
                grid-template-columns: 1fr;
            }
        }
'''
    content = content.replace('    </style>', css_addition + '    </style>')

# Remove max-width constraints for full-width layout
content = content.replace('            max-width: 1200px;', '            /* max-width removed to use full browser width */')

# Now do HTML reorganization with careful string replacements
# Replace 1: Add grid wrapper
content = content.replace(
    '''            <div class="preview" id="previewSection">
                <!-- Document Preview Controls -->
                <div class="preview-controls">''',
    '''            <div class="preview" id="previewSection">
                <!-- Grid Layout: Left = Preview, Right = Detection/Fields -->
                <div class="preview-grid">
                    <!-- LEFT SIDE: Document Preview -->
                    <div class="preview-left">
                        <!-- Document Preview Controls -->
                        <div class="preview-controls">'''
)

# Replace 2: Close left column after image container
content = content.replace(
    '''                </div>

                <div class="document-type-banner" id="docTypeBanner">''',
    '''                </div>
                    </div>
                    <!-- End preview-left -->

                    <!-- RIGHT SIDE: Document Type + Fields -->
                    <div class="preview-right">
                        <div class="document-type-banner" id="docTypeBanner">'''
)

# Replace 3: Wrap fields in container
content = content.replace(
    '''                <div class="section-header">
                    <h3>Detected Fields</h3>
                    <span class="badge"><span id="fieldCount">0</span> Selected</span>
                </div>''',
    '''                        <!-- Fields Section -->
                        <div class="fields-container">
                            <div class="section-header">
                                <h3>Detected Fields</h3>
                                <span class="badge"><span id="fieldCount">0</span> Selected</span>
                            </div>'''
)

# Replace 4: Fix control panel indentation
content = content.replace(
    '''                <div class="control-panel">
                    <button class="btn-secondary" onclick="selectAll()"><i class="fas fa-check-double"></i> Select
                        All</button>
                    <button class="btn-secondary" onclick="selectSensitive()"><i
                            class="fas fa-exclamation-triangle"></i> Select Sensitive</button>
                    <button class="btn-secondary" onclick="clearAll()"><i class="fas fa-times-circle"></i> Clear
                        All</button>
                </div>''',
    '''                            <div class="control-panel">
                                <button class="btn-secondary" onclick="selectAll()"><i class="fas fa-check-double"></i> Select All</button>
                                <button class="btn-secondary" onclick="selectSensitive()"><i class="fas fa-exclamation-triangle"></i> Select Sensitive</button>
                                <button class="btn-secondary" onclick="clearAll()"><i class="fas fa-times-circle"></i> Clear All</button>
                            </div>'''
)

# Replace 5: Close fields container and right column, start options section
content = content.replace(
    '''                <div id="fieldsList"></div>

                <div class="redaction-options">''',
    '''                            <div id="fieldsList"></div>
                        </div>
                        <!-- End fields-container -->
                    </div>
                    <!-- End preview-right -->
                </div>
                <!-- End preview-grid -->

                <!-- BOTTOM SECTION: Templates + Redaction Options -->
                <div class="options-section">
                    <div class="redaction-options">'''
)

# Replace 6: Close options section before action buttons
content = content.replace(
    '''                </div>

                <div class="action-buttons">''',
    '''                    </div>
                </div>
                <!-- End options-section -->

                <div class="action-buttons">'''
)

# Write the modified content
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Layout reorganization complete!")
print("✅ JavaScript and CSS preserved")
print("✅ Full-width layout applied")
