How to create a virtual environment in VSCode python using terminal:

       if  windows/linux machine: python -m venv .venv

How to activate the virtual environment:

        if  windows machine: .venv\Scripts\activate
        
        if linux machine:    source .venv/bin/activate

How to install packages in python:

    if  windows/linux machine: pip install package_name

    # Example: 
        pip install streamlit
        pip install pandas
        pip install langchain
        pip install llama_index 


Task

Summary

Description

Criteria for Done

Acceptance Criteria

Correctly identify visible elements

Enhance element detection to validate visibility and presence

Improve the AI engine to ensure that only elements that are actually present in the DOM and visible to the user are considered for test generation. This avoids false positives from hidden or detached elements.

- Visibility check logic implemented (e.g., display, opacity, viewport)- DOM presence validation added- Existing crawler updated to filter non-visible elements- Unit tests added for visibility scenarios

- Given a webpage, when elements are scanned, then only visible and interactable elements are identified- Hidden elements (display:none, offscreen, etc.) are ignored- Generated test cases do not include non-visible elements

Provide context via user story

Allow manual input of user story for better test generation

Introduce a feature where users can provide a user story or context to guide AI-based test case generation, improving relevance and coverage.

- UI/input field for user story added- AI model updated to incorporate context- Context stored and linked to generated outputs- Tested with multiple user story inputs

Given a user story is provided, when test cases are Generated, then they reflect the intended business flow- Without user story, system falls back to default behavior

Manual intervention for element selection

Enable user override for element selection and locator strategy

Provide an option for users to manually select elements and define or override locator strategies (XPath, CSS, etc.) when AI-generated locators are incorrect or ambiguous.

- UI for manual element selection implemented- Locator editing capability added- Changes persist in generated scripts- Audit/log of manual overrides maintained

- Given incorrect AI locator, when user overrides it, then updated locator is used in scripts- Manual selections are reflected in regenerated test cases

Crawl site with login

Enable authenticated crawling of web applications

Add support for crawling websites that require login by allowing users to provide credentials or session-based authentication.

- Login handling mechanism implemented (form/session/token)- Secure credential handling ensured- Crawl continues post-login- Tested across common auth flows

- Given valid credentials, when crawling starts, then system logs in and crawls protected pages- Unauthorized access is handled gracefully

Multiple test executions

Enable repeated execution of generated test scripts

Allow users to run generated test scripts multiple times (on-demand or scheduled), instead of a single execution.

- Execution trigger mechanism implemented- Support for rerun and scheduling added- Execution history/logs maintained- Configurable run parameters supported

- Given a test script, when user triggers execution multiple times, then it runs successfully each time- Execution results are stored and viewable

Mobile application testing

Extend support for mobile app test automation

Enhance the platform to support mobile application testing (Android/iOS), including UI element detection and test generation.

- Mobile platform integration (e.g., Appium) added- Element detection for mobile UI implemented- Test generation supports mobile flows- Tested on sample mobile apps

- Given a mobile app, when processed, then test cases and scripts are generated for mobile UI- Scripts can run on mobile environments

Tool integrations

Integrate with TestRail, Figma, Hydra, and  boilerplate

Build integrations with existing tools used by teams to improve workflow efficiency, traceability, and adoption.

- APIs for TestRail, Figma, Hydra, boilerplate integrated- Data mapping defined for each tool- Authentication handled securely- Integration tested end-to-end

- Given integration is configured, when tests are generated/executed, then results sync with TestRail- Design references from Figma are accessible- Hydra and boilerplate workflows are supported

Task

Summary

Description

Criteria for Done

Acceptance Criteria

CVT & Smart Capture Integration

Integrate CVT and Smart Capture for contextual UI validation

Integrate CVT and Smart Capture capabilities to validate UI changes and contextual correctness, enabling the system to generate a more complete and reliable automation package.

- CVT and Smart Capture APIs integrated- UI change detection mechanism implemented- Context validation logic added- End-to-end flow tested

- Given a UI change, when processed, then system detects and validates changes accurately- Generated test assets reflect updated UI context

Manual intervention at all stages

Enable manual intervention across all stages of test generation

Provide users the ability to intervene and modify outputs at each stage of the pipeline (locator detection, test case generation, feature file creation, script generation).

- Intervention points added at each stage- Editable UI/components implemented- Changes persist across pipeline- Logging/audit maintained

- Given any stage output, when user modifies it, then downstream outputs reflect the changes- Users can update scripts with minimal effort

Negative & edge case generation

Enhance AI to generate negative, boundary, and edge test scenarios

Improve AI logic to include negative scenarios, boundary conditions, and edge cases during test generation for better coverage and robustness.

- AI model updated with negative/edge case logic- Scenario templates enhanced- Test coverage validation added- Tested across multiple use cases

- Given a feature, when test cases are generated, then positive, negative, and edge cases are included- Boundary conditions are explicitly covered

Parallel test execution

Enable parallel execution of test cases

Implement capability to run multiple test cases in parallel to reduce execution time and improve efficiency.

- Parallel execution framework implemented- Resource allocation handled- Thread-safe execution ensured- Execution reporting updated

- Given multiple test cases, when executed, then they run in parallel without conflicts- Execution time is reduced compared to sequential runs

Cross-browser testing

Support execution across multiple browsers

Enable test execution across different browsers (e.g., Chrome, Firefox, Safari, Edge) to ensure compatibility.

- Browser configuration support added- Integration with execution environments (e.g., BrowserStack/local)- Cross-browser test validation implemented

- Given a test script, when executed across browsers, then results are consistent- Browser-specific issues are identified

Device-specific testing

Enable testing across different device configurations

Add capability to run tests on different devices (desktop, tablet, mobile) and configurations (OS, resolution, etc.).

- Device configuration support added- Integration with device farms/emulators- Responsive validation implemented

- Given a test case, when executed on different devices, then UI and functionality are validated correctly- Device-specific issues are captured

Schedule test execution

Enable scheduling of automated test runs

Provide functionality to schedule test executions at predefined times or intervals for continuous testing.

- Scheduler component implemented- UI for scheduling added- Time/interval configuration supported- Execution logs maintained

- Given a schedule is set, when the scheduled time occurs, then tests execute automatically- Results are recorded and accessible

Task

Summary

Description

Criteria for Done

Acceptance Criteria

Report generation clarity & sharing

Improve test report clarity and enable report sharing

Enhance reporting to clearly differentiate between test generation summary and actual execution report, and add sharing/export capabilities to avoid confusion.

- Separate report types defined (generation vs execution)- UI updated with clear labels and sections- Report export/share options added (PDF/URL/email)- Tested for usability and clarity

- Given script generation is completed, when report is viewed, then it is clearly labeled as “Generation Summary”- Given test execution is completed, when report is viewed, then it shows execution results distinctly- Users can share/export reports easily

Compare executions across environments

Enable comparison of test results across environments

Provide capability to compare test execution results for the same URL across different environments (e.g., QA, staging, prod).

- Comparison module implemented- Environment tagging supported- UI for side-by-side comparison added- Data consistency validated

- Given multiple executions for same site, when comparison is triggered, then differences in results are displayed clearly- Metrics like pass/fail/duration are comparable

Framework selection for Python generation

Allow framework selection during script generation

Provide users an option to choose the automation framework (e.g., PyTest, Unittest, Behave) when generating Python test scripts.

- Framework selection UI added- Script generation logic updated for each framework- Templates created for supported frameworks- Tested for all options

- Given a framework is selected, when script is generated, then it follows the selected framework structure- Switching frameworks produces correct script format

Browser selection for native execution

Enable browser selection for native test execution

Add option for users to choose browser (Chrome, Firefox, etc.) when running tests locally (native execution).

- Browser selection option added in UI/CLI- Execution engine updated to support browsers- Tested across supported browsers

- Given a browser is selected, when test runs natively, then it executes in chosen browser- Invalid selections are handled gracefully

Optional headless mode

Provide optional headless execution mode

Allow users to enable or disable headless mode during execution for flexibility in debugging vs performance.

- Headless toggle implemented- Execution engine updated- Tested for both modes

- Given headless mode is enabled, when tests run, then no browser UI is displayed- Given disabled, browser UI is visible

Functional grouping of elements

Improve understanding of functional workflows vs single-page features

Enhance AI to understand relationships between elements and group them into meaningful functional flows rather than treating entire page as one feature.

- AI logic updated for flow detection- Element relationship mapping implemented- Feature grouping logic enhanced- Validated with real-world flows

- Given a webpage, when test cases are generated, then they reflect user workflows (e.g., login, checkout)- Features are logically grouped, not page-based only

Data-driven testing support

Enable data-driven testing capabilities

Introduce support for running the same test with multiple data inputs (e.g., parameterization, datasets).

- Data input mechanism added (CSV/JSON/UI)- Script generation supports parameterization- Execution engine supports iteration- Tested with multiple datasets

- Given a dataset, when test runs, then it executes for all data variations- Results are tracked per dataset

Validation for manual edits

Provide validation and templates for manual edits in scripts/feature files

Add validation, guidelines, and templates to ensure that manually edited feature files and scripts follow correct syntax and structure.

- Syntax validation implemented- Error detection and feedback added- Predefined templates provided- Documentation/guidelines added

- Given a user edits a script, when saved, then syntax errors are detected and shown- Valid edits pass validation- Users can refer to templates for correct format

Task

Summary

Description

Criteria for Done

Acceptance Criteria

Unified framework for script execution

Introduce a robust framework to integrate multiple scripts into cohesive workflows

Current scripts run independently, which limits testing of complex, end-to-end functionalities. Introduce a unified framework to orchestrate multiple scripts and enable them to function as a single integrated flow.

- Framework for test orchestration implemented- Support for chaining scripts added- Dependency handling between scripts enabled- End-to-end flow testing validated

- Given multiple scripts, when executed, then they run as a connected workflow- Dependencies between features are handled correctly- Complex user journeys can be tested end-to-end

CI/CD integration

Integrate with CI/CD pipelines for automated test execution

Enable integration with CI/CD tools (e.g., Jenkins, GitHub Actions, Azure DevOps) to allow automated triggering of tests during build and deployment cycles.

- CI/CD integration APIs/hooks implemented- Documentation for setup provided- Trigger mechanisms (webhook/manual) added- Tested in at least one CI/CD tool

- Given a pipeline is configured, when a build is triggered, then tests run automatically- Results are accessible within pipeline logs

Team collaboration features

Enable team-level collaboration and access control

Provide features for multiple users/teams to collaborate within the platform, including role-based access, sharing, and visibility controls.

- User roles and permissions implemented- Project sharing capability added- Activity logs maintained- UI supports multi-user workflows

- Given multiple users, when accessing a project, then permissions are enforced- Teams can collaborate without conflicts

Multiple environment URL support

Allow multiple environment URLs within a single project

Provide the ability to configure multiple URLs (e.g., dev, QA, staging, prod) under the same project for easier environment management.

- Environment configuration module added- URL mapping supported- UI for environment selection implemented- Tested across environments

- Given multiple environment URLs, when tests are run, then selected environment is used- Users can switch environments easily

Execution comparison within project

Enable comparison of test executions within the same project

Allow users to compare different test executions (runs) within a project to track regressions and improvements over time.

- Execution history tracking implemented- Comparison UI added- Metrics normalization ensured- Tested with multiple runs

- Given multiple executions, when compared, then differences in pass/fail and duration are shown clearly- Trends are identifiable

Error classification and insights

Add intelligent error grouping and classification

Introduce features to identify and categorize errors into unique, similar, and new errors to improve debugging and reporting efficiency.

- Error classification logic implemented- Similarity detection added- UI for grouped errors created- Logs enriched with metadata

- Given execution results, when errors occur, then they are grouped as unique/similar/new- Users can quickly identify recurring issues

Improve UI navigation performance

Optimize navigation speed across application sections

Address performance issues in navigating between Projects → Test Suites → Test Scripts to provide a faster and smoother user experience.

- Performance bottlenecks identified- UI/API optimizations implemented- Lazy loading/caching added where needed- Performance benchmarks improved

- Given user navigation, when switching sections, then load time is significantly reduced- Navigation feels responsive and smooth

Optimize test suite creation time

Improve efficiency of test suite generation

Reduce the time taken to create test suites, especially for multiple URLs, by optimizing crawling, processing, and generation logic.

- Code optimization implemented- Parallel processing where applicable- Performance metrics improved- Tested with multiple URLs

- Given multiple URLs, when test suite is created, then time taken is reduced compared to baseline- No degradation in output quality

Task

Summary

Description

Criteria for Done

Acceptance Criteria

Improve link crawling & flow

Enhance link discovery and maintain logical flow during crawling

Current crawling misses some links and does not follow a systematic flow, leading to incomplete coverage and improper feature segregation. Improve crawling logic to ensure complete and structured traversal of the website.

- Improved crawling algorithm implemented- Duplicate/missed link detection added- Logical flow sequencing ensured- Feature mapping aligned with crawl structure

- Given a website, when crawling is executed, then all accessible links are captured- Links are processed in a logical flow- Feature files are generated in a structured manner

Pre-generation filtering control

Allow users to refine/exclude elements before test generation

Provide users the ability to filter out unwanted elements or sections before generating test cases to improve relevance and reduce noise.

- UI for element selection/exclusion added- Filtering logic implemented- Preview before generation supported- Changes persisted

- Given a set of detected elements, when user excludes some, then they are not used in generation- Generated outputs reflect only selected elements

Scenario naming standardization

Standardize and clean scenario naming conventions

Fix improper scenario naming (e.g., raw SVG/XML text being used) by implementing meaningful naming conventions based on element context and purpose.

- Naming rules defined and implemented- Text sanitization logic added- Edge cases (icons, SVGs, hidden labels) handled- Tested across scenarios

- Given an element, when scenario is generated, then name is readable and meaningful- Raw code/text is not used in scenario names

Pop-up, modal & alert handling

Add support for pop-ups, modals, and alerts

Extend automation capabilities to handle dynamic UI elements such as pop-ups, modals, and browser alerts which are currently unsupported.

- Detection logic for pop-ups/modals implemented- Script handling added (accept/dismiss/interact)- Test cases include such scenarios- Tested across common patterns

- Given a pop-up/modal/alert, when encountered, then system handles it correctly- Test scripts include appropriate handling steps

Feature file quality improvement

Generate well-structured and standardized feature files

Improve feature file generation to ensure clarity, avoid repetition, and maintain consistency and readability across all scenarios.

- Standard template defined- Reusability of steps implemented- Redundant steps eliminated- Output validated for structure

- Given generated feature files, when reviewed, then they are clean, consistent, and non-repetitive- All test cases are correctly translated into feature format

Reduce manual effort in validation

Improve output quality to minimize manual corrections

Enhance AI accuracy and generation logic so that outputs are closer to “ready-to-use” and require minimal manual intervention.

- Accuracy improvements implemented- Validation checks added- Output quality benchmarks defined- Tested against baseline

- Given generated outputs, when reviewed, then minimal manual fixes are required- Majority of test cases are usable directly

Reliable test case to script conversion

Ensure accurate and complete conversion from test cases to scripts

Address gaps where not all scenarios reliably convert into scripts and improve handling of execution aspects like waits, scrolling, and synchronization.

- Conversion logic improved- Wait/sync/scroll handling added- Coverage validation implemented- Tested for different scenarios

- Given a test case, when converted, then corresponding script is generated correctly- Scripts include proper waits and synchronization handling

Preview & validation before script generation

Add validation and preview for test cases before script generation

Introduce a preview and validation layer for both generated and manually edited test cases before converting them into scripts.

- Preview UI implemented- Validation rules added- Error highlighting and suggestions provided- Integration with generation pipeline completed

- Given test cases (new or edited), when previewed, then issues are highlighted- Only validated test cases proceed to script generation

Task

Summary

Description

Criteria for Done

Acceptance Criteria

Improve test case to script conversion transparency

Provide clear visibility into test case to script conversion outcomes

Currently, only a subset of generated test cases are converted into scripts, with no clear explanation for failures. Enhance the system to provide detailed insights into conversion success and failure.

- Conversion tracking implemented for each test case- Failure reason logging added- UI updated to display conversion status per test case- Metrics (total, success, failed) clearly shown

- Given multiple test cases, when conversion occurs, then each test case shows success/failure status- Failed cases include a clear reason for failure

Detailed failure reasoning for conversion issues

Add detailed error reporting for failed script generation

Improve error handling to provide actionable and specific reasons for why certain test cases fail during script generation.

- Error classification implemented- Meaningful error messages added- Logs accessible via UI- Common failure scenarios documented

- Given a failed conversion, when user views details, then a clear and actionable error message is shown- Users can identify root cause without ambiguity

Scenario-wise code view

Provide scenario-level code visibility

Currently, code is shown as a combined script, making debugging difficult. Introduce scenario-wise code separation for better analysis.

- Code generation split per scenario- UI updated to display scenario-level scripts- Navigation between scenarios enabled- Tested for multiple scenarios

- Given multiple scenarios, when viewing code, then each scenario has its own script section- Users can easily debug individual scenarios

Align scenario details with feature files

Ensure scenario-level details match Gherkin format

Scenario-level details currently show step plans that do not align with Gherkin or feature files. Update system to ensure consistency and traceability.

- Scenario representation updated to Gherkin format- Mapping between feature file and scenario view implemented- Validation checks added

- Given a scenario, when viewed, then it matches the corresponding feature file in Gherkin format- Users can trace steps directly

Clarify execution vs evaluation status

Clearly differentiate between test evaluation and actual execution

Current reporting shows mismatches (expected vs actual) even before execution, causing confusion. Add clear distinction between evaluation phase and execution phase.

- Execution state tracking implemented- Status labels (evaluated vs executed) added- Reporting logic updated- UI updated for clarity

- Given a test case, when viewed, then it clearly indicates whether it was executed or just evaluated- No misleading result is shown before execution

Improve success result transparency

Provide detailed basis for successful execution results

Current success messages are vague and do not explain how results were derived. Improve transparency in reporting successful executions.

- Detailed result logging implemented- Step-level execution logs added- Evidence (logs/screenshots) captured- UI updated to show details

- Given a successful execution, when viewed, then detailed logs and results are available- Users understand how success was determined

Ensure consistency between plan, feature, and report

Maintain consistency across generated artifacts

Some steps in plans or reports do not match feature files, causing confusion. Ensure consistency across all generated outputs.

- Cross-validation logic implemented- Synchronization between artifacts ensured- Discrepancy detection added- Tested across flows

- Given a feature file, when plan/report is generated, then all steps match exactly- No mismatch between artifacts

Task

Summary

Description

Criteria for Done

Acceptance Criteria

Improve end-to-end test case conversion reliability

Strengthen logic to ensure all test cases convert into executable scripts

Enhance the conversion engine to reliably transform all generated test cases into executable scripts, minimizing failures and ensuring completeness.

- Conversion logic improved- Edge case handling added- Retry/fallback mechanisms implemented- Conversion success rate benchmarked and improved

- Given generated test cases, when conversion runs, then majority (ideally all) are successfully converted- Failures (if any) are minimal and justified

Execution logs with evidence

Provide detailed execution logs with proof for issue tracking

Enhance logging to include step-level logs, timestamps, screenshots, and other artifacts to help identify when and where failures occur.

- Step-level logging implemented- Screenshot/video capture added- Timestamp tracking enabled- Logs accessible via UI

- Given a test execution, when failure occurs, then logs clearly show the step and time of failure- Evidence (screenshots/logs) is available

Dynamic UI handling improvements

Improve support for dynamic UI elements and behaviors

Strengthen automation to handle dynamic front-end elements like pop-ups, overlays, and conditional UI components more reliably.

- Detection logic enhanced- Handling strategies implemented (waits, retries, conditions)- Tested across dynamic UI cases

- Given dynamic elements, when encountered, then scripts handle them correctly- Tests do not fail due to timing or conditional rendering issues

Secure secrets management

Implement centralized and secure secrets management

Refactor code to remove hardcoded sensitive data and use a centralized secrets manager for improved security and compliance.

- Secrets manager integrated- Hardcoded values removed- Secure access mechanisms implemented- Security testing completed

- Given sensitive data usage, when system runs, then no secrets are hardcoded- Secrets are securely retrieved from centralized manager

Strengthen CORS configuration

Restrict CORS access to approved domains only

Improve security by tightening CORS policies to allow only whitelisted domains and prevent unauthorized access.

- CORS policy updated- Whitelisting mechanism implemented- Security validation completed

- Given API requests, when origin is unapproved, then request is blocked- Approved domains function without issues

CSS optimization and cleanup

Refactor CSS for maintainability and performance

Clean up redundant and conflicting CSS rules to reduce overrides and improve long-term maintainability of UI components.

- Redundant styles removed- CSS structure standardized- Performance improvements validated- UI regression testing completed

- Given UI components, when rendered, then styles are consistent and optimized- No unnecessary overrides exist

Code-level findings implementation

Address identified code-level improvements

Implement improvements identified during code analysis to enhance performance, maintainability, and reliability.

- All findings documented- Fixes implemented and reviewed- Code quality checks passed- Regression testing completed

- Given code review findings, when implemented, then issues are resolved- No regression introduced

Optimize LLM API usage

Reduce number of LLM calls per request

Optimize architecture to minimize LLM API calls per request, improving performance and reducing cost.

- Call optimization logic implemented- Caching/batching strategies added- Performance benchmarks improved

- Given a request, when processed, then fewer LLM calls are made compared to baseline- Output quality remains consistent

Evaluate cost-efficient LLM alternatives

Assess alternative LLM models for cost optimization

Evaluate and integrate alternative LLM models that provide similar output quality at lower cost.

- Alternative models identified and tested- Benchmarking (cost vs quality) completed- Integration feasibility assessed

- Given multiple models, when evaluated, then cost vs performance comparison is available- Recommended model meets quality standards

Documentation and user guidance

Provide comprehensive documentation and in-product guidance

Add clear documentation, inline help, and usage guidance to improve usability and adoption of the platform.

- Documentation created/updated- Inline help/tooltips added- User guides and examples included- Reviewed for clarity

- Given a user interacts with the system, when guidance is needed, then help/documentation is easily accessible- Users can understand features without external support

Regards,
Deepak
