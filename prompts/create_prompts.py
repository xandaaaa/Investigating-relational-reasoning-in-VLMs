import json
import random
from typing import List, Dict

class QuestionGenerator:
    def __init__(self, json_data: Dict):
        self.data = json_data
        self.image_id = json_data['image_id']
        self.image_filename = json_data['image_filename']
        self.entities = json_data['entities']
        self.relations = json_data['relations']
        self.shapes = ['circle', 'square', 'rectangle', 'triangle']
        self.colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'purple', 'orange']
        
    def format_question(self, question: str, options: List[str], answer: str) -> Dict:
        """Format question with options and answer"""

        option_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
        formatted_options = ' '.join([f"{option_labels[i]} {opt}" for i, opt in enumerate(options)])

        formatted_question = f"{question}? Here are your options: {formatted_options} Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'."
        return {
            'question': formatted_question,
            'answer': answer
        }
    
    def generate_count_question(self) -> Dict:
        """Generate question about number of shapes in image"""
        actual_count = len(self.entities)
        
        # Generate wrong options (with 4 options in total 3 wrong, 1 correct)
        wrong_counts = [i for i in range(0, 9) if i != actual_count]
        options_counts = random.sample(wrong_counts, 3)
        options_counts.append(actual_count)
        random.shuffle(options_counts)
        
        # stringify options and get answer
        options = [str(count) for count in options_counts]
        correct_idx = options.index(str(actual_count))
        answer = ['a)', 'b)', 'c)', 'd)'][correct_idx]
        
        return self.format_question(
            "How many shapes are in this image",
            options,
            answer
        )
    
    def generate_recognition_questions(self) -> List[Dict]:
        """Generate recognition questions about shapes, 1 QUESTION EACH"""
        questions = []
        
        # Get shapes and colors that are in the image
        shapes_in_image = {entity['shape'] for entity in self.entities}
        colors_in_image = {entity['color'] for entity in self.entities}
        shape_color_pairs = {(entity['shape'], entity['color']) for entity in self.entities}
        
        # True cases (Shape)
        asked_shapes = set()
        for entity in self.entities[:1]:
            actual_shape = entity['shape']
            
            if actual_shape not in asked_shapes:
                questions.append(self.format_question(
                    f"Does this image have a {actual_shape} shape?",
                    ['a) Yes', 'b) No'],
                    'a)'
                ))
            asked_shapes.add(actual_shape)
        
        # False cases (Shape)
        shapes_not_in_image = [s for s in self.shapes if s not in shapes_in_image]
        if shapes_not_in_image:
            wrong_shape = random.choice(shapes_not_in_image)
            questions.append(self.format_question(
                f"Does this image have a {wrong_shape}",
                ['Yes', 'No'],
                'b)'
            ))
        
        # True cases (Color)
        asked_color = set()
        for entity in self.entities[:1]:
            actual_color = entity['color']

            if actual_color not in asked_color:
                questions.append(self.format_question(
                    f"Does this image have a {actual_color} shape",
                    ['Yes', 'No'],
                    'a)'
                ))
            asked_color.add(actual_color)
        
        # False cases (Color)
        colors_not_in_image = [c for c in self.colors if c not in colors_in_image]
        if colors_not_in_image:
            wrong_color = random.choice(colors_not_in_image)
            questions.append(self.format_question(
                f"Does this image have a {wrong_color} shape",
                ['Yes', 'No'],
                'b)'
            ))
        
        # True cases (Shape + Color)
        for entity in self.entities[:1]:
            shape = entity['shape']
            color = entity['color']
            questions.append(self.format_question(
                f"Does this image have a {color} {shape}",
                ['Yes', 'No'],
                'a)'
            ))
        
        # False cases (Shape + Color)
        wrong_pair_found = False
        for shape in self.shapes:
            for color in self.colors:
                if (shape, color) not in shape_color_pairs:
                    questions.append(self.format_question(
                        f"Does this image have a {color} {shape}",
                        ['Yes', 'No'],
                        'b)'
                    ))
                    wrong_pair_found = True
                    break
            if wrong_pair_found:
                break
        
        return questions
    
    def generate_implicit_questions(self) -> List[Dict]:
        """Generate questions about implicit spatial relationships (no arrows)"""
        questions = []

        # Get all implicit relations
        implicit_relations = [relation for relation in self.relations if not relation['explicit']]
        random.shuffle(implicit_relations)

        # Limit to 1 question
        for relation in implicit_relations[:1]:
            
            # get correct subject and object relation
            subject = self.entities[relation['subject_id']]
            obj = self.entities[relation['object_id']]
            rel = relation['relation']
            
            options = [
                f"above the {subject['color']} {subject['shape']}",
                f"below the {subject['color']} {subject['shape']}",
                f"to the left of the {subject['color']} {subject['shape']}",
                f"to the right of the {subject['color']} {subject['shape']}"
            ]
            
            # Map relation to option
            rel_to_option = {
                'above': 0,
                'below': 1,
                'left_of': 2,
                'right_of': 3
            }
            
            correct_idx = rel_to_option[rel]
            answer = ['a)', 'b)', 'c)', 'd)'][correct_idx]
            
            questions.append(self.format_question(
                f"What is the position of the {obj['color']} {obj['shape']} with respect to the {subject['color']} {subject['shape']}",
                options,
                answer
            ))
        
        return questions
    
    def generate_explicit_questions(self) -> List[Dict]:
        """Generate questions about explicit relationships (arrows)"""

        # CONNECTION ONLY
        questions = []

        explicit_relations = [r for r in self.relations if r['explicit']]

        if not explicit_relations:
            return questions

        # Build pairs of connected objects
        connected_pairs = []
        for relation in explicit_relations:
            subject = self.entities[relation['subject_id']]
            obj = self.entities[relation['object_id']]
            pair_text = f"the {subject['color']} {subject['shape']} with the {obj['color']} {obj['shape']}"
            connected_pairs.append(pair_text)

        connected_pairs.sort()
        correct_option = " and ".join(connected_pairs)
        correct_relation_set = {(r['subject_id'], r['object_id']) for r in explicit_relations}

        # Try to create 3 wrong options
        all_entities = [e for e in self.entities]
        wrong_answers = []
        attempts = 0

        # use colors and shapes not in image if we cannot build wrong relations in image
        use_random = False
        while len(wrong_answers) < 3 and attempts < 200:
            attempts += 1

            if not use_random and len(all_entities) >= 2:
                # Generate same number of random pairs
                wrong_pairs = []
                wrong_relation_set = set()
                used_pairs_in_option = set()
                
                for _ in range(len(connected_pairs)):
                    pair_attempts = 0
                    # Find wrong pairs from existing shapes in image
                    while pair_attempts < 20:
                        e1, e2 = random.sample(all_entities, 2)
                        pair_key = (e1['id'], e2['id'])
                        
                        if pair_key not in used_pairs_in_option:
                            wrong_pairs.append(f"the {e1['color']} {e1['shape']} with the {e2['color']} {e2['shape']}")
                            wrong_relation_set.add(pair_key)
                            used_pairs_in_option.add(pair_key)
                            break
                        pair_attempts += 1
                    
                    if pair_attempts >= 20:
                        break
                
                # Only use this set of pairs if theyre not the correct one and was not a wrong option that was appended before already
                if len(wrong_pairs) == len(connected_pairs) and wrong_relation_set != correct_relation_set:
                    wrong_pairs.sort()
                    wrong_option = " and ".join(wrong_pairs)
                    if wrong_option != correct_option and wrong_option not in wrong_answers:
                        wrong_answers.append(wrong_option)
                
                if attempts > 50:
                    use_random = True
            else:
                # Use random colors and shapes
                wrong_pairs = []
                used_pairs_in_option = set()
                
                for _ in range(len(connected_pairs)):
                    pair_attempts = 0
                    while pair_attempts < 20:
                        color1 = random.choice(self.colors)
                        shape1 = random.choice(self.shapes)
                        color2 = random.choice(self.colors)
                        shape2 = random.choice(self.shapes)
                        pair_text = f"the {color1} {shape1} with the {color2} {shape2}"
                        
                        if pair_text not in used_pairs_in_option:
                            wrong_pairs.append(pair_text)
                            used_pairs_in_option.add(pair_text)
                            break
                        pair_attempts += 1
                    
                    if pair_attempts >= 20:
                        break
                
                if len(wrong_pairs) == len(connected_pairs):
                    wrong_pairs.sort()
                    wrong_option = " and ".join(wrong_pairs)
                    
                    if wrong_option != correct_option and wrong_option not in wrong_answers:
                        wrong_answers.append(wrong_option)

        options = [correct_option] + wrong_answers
        random.shuffle(options)

        correct_idx = next(i for i, opt in enumerate(options) if opt == correct_option)
        answer = ['a)', 'b)', 'c)', 'd)'][correct_idx]

        questions.append(self.format_question(
            "Which objects are connected",
            options,
            answer
        ))
        
        # ARROW BASED
        random.shuffle(explicit_relations)
        for relation in explicit_relations[:1]:
            subject = self.entities[relation['subject_id']]
            obj = self.entities[relation['object_id']]
            
            correct_option = f"from the {subject['color']} {subject['shape']} to the {obj['color']} {obj['shape']}"
            
            # Try to create up to 3 wrong options
            all_entities = [e for e in self.entities]
            wrong_answers = []
            attempts = 0

            # use colors and shapes not in image if we cannot build wrong relations in image
            use_random = False
            while len(wrong_answers) < 3 and attempts < 200:
                attempts += 1
                
                # First try sampling from existing entities
                if not use_random and len(all_entities) >= 2:
                    wrong_entity1, wrong_entity2 = random.sample(all_entities, 2)
                    wrong_option = f"from the {wrong_entity1['color']} {wrong_entity1['shape']} to the {wrong_entity2['color']} {wrong_entity2['shape']}"
                    
                    if wrong_option != correct_option and wrong_option not in wrong_answers:
                        wrong_answers.append(wrong_option)
                    
                    # switch to random shapes and colors
                    if attempts > 50:
                        use_random = True
                else:
                    color1 = random.choice(self.colors)
                    shape1 = random.choice(self.shapes)
                    color2 = random.choice(self.colors)
                    shape2 = random.choice(self.shapes)
                    
                    wrong_option = f"from the {color1} {shape1} to the {color2} {shape2}"
                    
                    if wrong_option != correct_option and wrong_option not in wrong_answers:
                        wrong_answers.append(wrong_option)
            
            options = [correct_option] + wrong_answers
            random.shuffle(options)
            
            correct_idx = next(i for i, opt in enumerate(options) if opt == correct_option)
            answer = ['a)', 'b)', 'c)', 'd)'][correct_idx]
            
            questions.append(self.format_question(
                f"Where does the arrow between the {subject['color']} {subject['shape']} and {obj['color']} {obj['shape']} point to",
                options,
                answer
            ))
        
        return questions
    
    def generate_all_questions(self) -> List[Dict]:
        """Generate all types of questions"""
        all_questions = []
        
        # Add count question
        all_questions.append(self.generate_count_question())
        
        # Add recognition questions
        all_questions.extend(self.generate_recognition_questions())
        
        # Add implicit relationship questions
        all_questions.extend(self.generate_implicit_questions())
        
        # Add explicit relationship questions
        all_questions.extend(self.generate_explicit_questions())
        
        return all_questions

def example_print_questions():
    """prints all queries for a json file (json_file_path) from the synthetic dataset"""

    # Example JSON data
    json_file_path = 'synthetic_dataset_generation/output/annotations/annotation_00001.json'

    # Load JSON
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)
    
    # Generate questions
    json_data = load_json(json_file_path)
    generator = QuestionGenerator(json_data)
    questions = generator.generate_all_questions()
    
    # Print results
    for i, q in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}:")
        print(f"Q: {q['question']}")
        print(f"A: {q['answer']}")

def main():
    pass

if __name__ == "__main__":
    # example_print_questions()
    pass