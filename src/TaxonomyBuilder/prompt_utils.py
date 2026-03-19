class PromptBuilder:
    def __init__(self, name, definition):
        """
        Args:
            name (str): e.g., "Task" or "Skill" or whatever!
            definition (str): e.g., "A task is a specific activity..."
        """
        self.name = name
        self.definition = definition
        self.examples = []

    def add_example(self, statements: list, description: str):
        """Adds a few-shot example. Max 3 allowed."""
        if len(self.examples) < 3:
            stmt_str = str(statements)
            self.examples.append({"stmt": stmt_str, "desc": description})
        else:
            print("Max few-shot examples already added! Use clear_examples to start over.")
        return self
    
    def clear_examples(self):
        """Resets few-shot examples"""
        self.examples = []

    def build(self, target_statements: list) -> str:
        """Assemble the final prompt string."""
        example_block = ""
        for i, ex in enumerate(self.examples):
            example_block += f"\nstatements: {ex['stmt']}\n\nOutput:::\nDescription: {ex['desc']}\n"

        target_str = str(target_statements)

        prompt = f"""You will be given a list of statements.
Your job is to produce a single sentence that summarizes these statements into a coherent {self.name} description.
{self.definition}
Avoid using generalizations like "various" and "across domains".
Answer simply with the generated description, nothing else is required.

Provide your feedback as follows:

Output:::
Description: (GENERATED {self.name.upper()} DESCRIPTION)

Here are some examples:
{example_block}
Now here are the actual statements.

statements: {target_str}

Output:::
Description: """
        return prompt.strip()
    
    def build_mid(self, target_statements: list) -> str:
        """Assemble prompt string for a middle level of taxonomy."""
        target_str = str(target_statements)

        prompt = f"""
You will be given a list of statements.
Your job is to produce a single, very brief descriptor that summarizes these statements into a coherent description.
Avoid general terms like "various" and "across domains" when producing the descriptors.
Answer simply with the generated descriptor, nothing else is required.

Provide your feedback as follows:

Output:::
Description: (GENERATED DESCRIPTION)

Here are some examples:

statements: ['Develop software or computer applications.',
   'Develop computer or information systems.',
   'Design computer modeling or simulation programs.',
   'Develop computer or online applications.',
   'Design integrated computer systems.',
   'Develop software or applications for scientific or technical use.',
   'Modify software programs to improve performance.',
   'Design websites or web applications.',
   'Design video game features or details.',
   'Design software applications.',
   'Design healthcare-related software applications.',
   'Apply information technology to solve business or other applied problems.']

Output:::
Description: Design computer or information systems or applications.

statements: ['Research engineering aspects of biological or chemical processes.',
   'Research engineering applications of emerging technologies.',
   'Research design or application of green technologies.',
   'Conduct research to gain information about products or processes.',
   'Research advanced engineering designs or applications.',
   'Research energy production, use, or conservation.',
   'Research human performance or health factors related to engineering or design activities.',
   'Research new technologies.',
   'Research product safety.',
   'Research methods to improve food products.']

Output:::
Description: Research technology designs or applications.

Now here are the actual statements.

statements: {target_str}

Output:::
Description: """
        return prompt.strip()
    
    def build_top(self, target_statements: list) -> str:
        """Assemble prompt string for the top level of taxonomy."""
        target_str = str(target_statements)

        prompt = f"""
You will be given a list of general descriptions.
Your job is to produce a single, extremely brief, and very generalized category title that summarizes these statements into a coherent phrase.
Answer simply with the generated title, nothing else is required.

Provide your feedback as follows:

Output:::
Title: (GENERATED TITLE)

Here are some examples:

statements: ['Monitor equipment operation.',
   'Monitor operations to ensure adequate performance.',
   'Monitor financial data or activities.',
   'Monitor traffic conditions.',
   'Monitor health conditions of humans or animals.',
   'Monitor individual behavior or performance.',
   'Monitor safety or security of work areas, facilities, or properties.',
   'Monitor external affairs, trends, or events.',
   'Monitor environmental conditions.',
   'Monitor operation of computer or information technologies.',
   'Monitor operations to ensure compliance with regulations or standards.']

Output:::
Title: Monitor Processes, Materials, or Surroundings

statements: ['Program computer systems or production equipment.',
   'Implement security measures for computer or information systems.',
   'Set up computer systems, networks, or other information systems.',
   'Resolve computer problems.',
   'Operate computer systems or computerized equipment.',
   'Process digital or online data.']

Output:::
Title: Interacting With Computers

Now here are the actual statements.

statements: {target_str}

Output:::
Title: """
        return prompt.strip()