import json
import random
from typing import List, Dict, Optional
from pathlib import Path

class QuestionGenerator:
    def __init__(self, json_data: Dict, masked_json_data: Optional[Dict] = None):
        self.data = json_data
        self.masked_data = masked_json_data
        self.image_id = json_data['image_id']
        self.image_filename = json_data['image_filename']
        self.entities = json_data['entities']
        self.relations = json_data['relations']
        
        # Masked data
        self.masked_entities = masked_json_data['entities'] if masked_json_data else []
        self.masked_relations = masked_json_data['relations'] if masked_json_data else []
        
        self.shapes = ['circle', 'square', 'rectangle', 'triangle']
        self.colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'purple', 'orange']
        
    def format_question(self, question: str, options: List[str], answer: str, query_type: str, answer_masked: str) -> Dict:
        """Format question with options and answer"""

        option_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']
        formatted_options = ' '.join([f"{option_labels[i]} {opt}" for i, opt in enumerate(options)])

        formatted_question = f"{question}? Here are your options: {formatted_options} Please only reply with the correct option, do not explain your reasoning. If no option is correct, reply with 'None'."
        return {
            'image_id': self.image_id,
            'image_filename': self.image_filename,
            'query_type': query_type,
            'query': formatted_question,
            'answer': answer,
            'answer_masked': answer_masked
        }
    
    def get_masked_count_answer(self, options: List[str]) -> str:

        if not self.masked_data:
            return ''
        
        actual_count = len(self.masked_entities)
        if str(actual_count) in options:
            return ['a)', 'b)', 'c)', 'd)'][options.index(str(actual_count))]
        return 'None'
    
    def get_masked_recognition_answer(self, query_type: str, shape: Optional[str] = None, color: Optional[str] = None) -> str:

        if not self.masked_data:
            return ''
        
        masked_shapes = {entity['shape'] for entity in self.masked_entities}
        masked_colors = {entity['color'] for entity in self.masked_entities}
        masked_pairs = {(entity['shape'], entity['color']) for entity in self.masked_entities}
        
        if query_type == "recognition_shape":
            return 'a)' if shape in masked_shapes else 'b)'
        elif query_type == "recognition_color":
            return 'a)' if color in masked_colors else 'b)'
        elif query_type == "recognition_shape_and_color":
            return 'a)' if (shape, color) in masked_pairs else 'b)'
        
        return 'None'
    
    def get_masked_implicit_answer(self, subject_id: int, object_id: int, answer: str) -> str:

        if not self.masked_data:
            return ''
        
        # Check if both entities still exist in masked version if not return None because there will be no spatial relationship between them
        masked_entity_ids = {entity['id'] for entity in self.masked_entities}
        if subject_id not in masked_entity_ids or object_id not in masked_entity_ids:
            return 'None'
        
        # Since we only have deletion implicit relations will be preserved if both entities still exist so we just return the same answer
        return answer

    def get_masked_explicit_connection_answer(self, options: List[str]) -> str:

        if not self.masked_data:
            return ''
        
        # Only consider: explicit=True with mask=False (or if mask doesnt exist exist)
        masked_explicit_relations = [
            r for r in self.masked_relations 
            if r["explicit"] and not r.get('masked', False)
        ]
        
        # No connected pairs return none
        if not masked_explicit_relations:
            return 'None'
        
        masked_entity_lookup = {e['id']: e for e in self.masked_entities}
        connected_pairs = []
        
        for relation in masked_explicit_relations:
            # Entity still has to exist
            if relation['subject_id'] in masked_entity_lookup and relation['object_id'] in masked_entity_lookup:
                subject = masked_entity_lookup[relation['subject_id']]
                obj = masked_entity_lookup[relation['object_id']]
                pair_text = f"the {subject['color']} {subject['shape']} with the {obj['color']} {obj['shape']}"
                connected_pairs.append(pair_text)
        
        if not connected_pairs:
            return 'None'
        
        connected_pairs.sort()
        correct_option = " and ".join(connected_pairs)
        
        # Check if this option exists in the original options
        if correct_option in options:
            return ['a)', 'b)', 'c)', 'd)'][options.index(correct_option)]
        
        return 'None'

    def get_masked_explicit_arrow_answer(self, subject_id: int, object_id: int, answer: str) -> str:

        if not self.masked_data:
            return ''
        
        # Check if both entities still exist
        masked_entity_lookup = {e['id']: e for e in self.masked_entities}
        if subject_id not in masked_entity_lookup or object_id not in masked_entity_lookup:
            return 'None'
        
        # If relation still exists (must be explicit and not masked) just return answer as they must be the same
        for relation in self.masked_relations:
            if (relation['subject_id'] == subject_id and 
                relation['object_id'] == object_id and 
                relation.get('explicit', False) and
                not relation.get('masked', False)):
                return answer
        
        return 'None'
    
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
        answer_masked = self.get_masked_count_answer(options)
        
        return self.format_question(
            "How many shapes are in this image",
            options,
            answer,
            "count",
            answer_masked
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
                answer_masked = self.get_masked_recognition_answer("recognition_shape", shape=actual_shape)
                questions.append(self.format_question(
                    f"Does this image have a {actual_shape} shape",
                    ['Yes', 'No'],
                    'a)',
                    "recognition_shape",
                    answer_masked
                ))
            asked_shapes.add(actual_shape)
        
        # False cases (Shape)
        shapes_not_in_image = [s for s in self.shapes if s not in shapes_in_image]
        if shapes_not_in_image:
            wrong_shape = random.choice(shapes_not_in_image)
            answer_masked = self.get_masked_recognition_answer("recognition_shape", shape=wrong_shape)
            questions.append(self.format_question(
                f"Does this image have a {wrong_shape}",
                ['Yes', 'No'],
                'b)',
                "recognition_shape",
                answer_masked
            ))
        
        # True cases (Color)
        asked_color = set()
        for entity in self.entities[:1]:
            actual_color = entity['color']

            if actual_color not in asked_color:
                answer_masked = self.get_masked_recognition_answer("recognition_color", color=actual_color)
                questions.append(self.format_question(
                    f"Does this image have a {actual_color} shape",
                    ['Yes', 'No'],
                    'a)',
                    "recognition_color",
                    answer_masked
                ))
            asked_color.add(actual_color)
        
        # False cases (Color)
        colors_not_in_image = [c for c in self.colors if c not in colors_in_image]
        if colors_not_in_image:
            wrong_color = random.choice(colors_not_in_image)
            answer_masked = self.get_masked_recognition_answer("recognition_color", color=wrong_color)
            questions.append(self.format_question(
                f"Does this image have a {wrong_color} shape",
                ['Yes', 'No'],
                'b)',
                "recognition_color",
                answer_masked
            ))
        
        # True cases (Shape + Color)
        for entity in self.entities[:1]:
            shape = entity['shape']
            color = entity['color']
            answer_masked = self.get_masked_recognition_answer("recognition_shape_and_color", shape=shape, color=color)
            questions.append(self.format_question(
                f"Does this image have a {color} {shape}",
                ['Yes', 'No'],
                'a)',
                "recognition_shape_and_color",
                answer_masked
            ))
        
        # False cases (Shape + Color)
        wrong_pair_found = False
        for shape in self.shapes:
            for color in self.colors:
                if (shape, color) not in shape_color_pairs:
                    answer_masked = self.get_masked_recognition_answer("recognition_shape_and_color", shape=shape, color=color)
                    questions.append(self.format_question(
                        f"Does this image have a {color} {shape}",
                        ['Yes', 'No'],
                        'b)',
                        "recognition_shape_and_color",
                        answer_masked
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
            answer_masked = self.get_masked_implicit_answer(relation['subject_id'], relation['object_id'], answer)
            
            questions.append(self.format_question(
                f"What is the position of the {obj['color']} {obj['shape']} with respect to the {subject['color']} {subject['shape']}",
                options,
                answer,
                "implicit_spatial",
                answer_masked
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
        correct_relation_set = set()
        for relation in explicit_relations:
            subject = self.entities[relation['subject_id']]
            obj = self.entities[relation['object_id']]
            pair_text = f"the {subject['color']} {subject['shape']} with the {obj['color']} {obj['shape']}"
            connected_pairs.append(pair_text)
            correct_relation_set.add((relation['subject_id'], relation['object_id']))

        connected_pairs.sort()
        correct_option = " and ".join(connected_pairs)

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
        answer_masked = self.get_masked_explicit_connection_answer(options)

        questions.append(self.format_question(
            "Which objects are connected",
            options,
            answer,
            "explicit_connection",
            answer_masked
        ))
        
        # ARROW BASED
        random.shuffle(explicit_relations)
        for relation in explicit_relations[:1]:
            subject = self.entities[relation['subject_id']]
            obj = self.entities[relation['object_id']]
            
            correct_option = f"from the {subject['color']} {subject['shape']} to the {obj['color']} {obj['shape']}"
            
            # Try to create 3 wrong options
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
            answer_masked = self.get_masked_explicit_arrow_answer(relation['subject_id'], relation['object_id'], answer)
            
            questions.append(self.format_question(
                f"Where does the arrow between the {subject['color']} {subject['shape']} and {obj['color']} {obj['shape']} point to",
                options,
                answer,
                "explicit_arrow_direction",
                answer_masked
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


def compile_json(annotations_folder: str, annotations_masked_folder: str, output_json: str):
    """
    Process all annotation JSON files in a folder and output to JSON
    
    Args:
        annotations_folder: Path to folder containing annotation JSON files
        annotations_masked_folder: Path to folder containing masked annotation JSON files
        output_json: Path to output JSON file
    """
    # Get all JSON files in the folder
    json_files = sorted(Path(annotations_folder).glob('*.json'))
    
    print(f"Found {len(json_files)} annotation files")
    
    # Group by image
    grouped_results = []
    
    for json_file in json_files:
        # Load original JSON
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        # Load masked JSON (same filename in masked folder)
        masked_filename = json_file.stem + '_masked.json'
        masked_file = Path(annotations_masked_folder) / masked_filename
        masked_json_data = None
        if masked_file.exists():
            with open(masked_file, 'r') as f:
                masked_json_data = json.load(f)
        else:
            print(f"No masked file found for: {masked_filename}")
        
        # Generate questions
        generator = QuestionGenerator(json_data, masked_json_data)
        questions = generator.generate_all_questions()
        
        # Create grouped structure
        image_data = {
            'image_id': json_data['image_id'],
            'image_filename': json_data['image_filename'],
            'questions': [
                {
                    'query_type': q['query_type'],
                    'query': q['query'],
                    'ground_truth': q['answer'],
                    'ground_truth_masked': q['answer_masked']
                }
                for q in questions
            ]
        }
        grouped_results.append(image_data)
    
    # Write to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(grouped_results, f, indent=2, ensure_ascii=False)
    
    total_questions = sum(len(img['questions']) for img in grouped_results)
    print(f"\nProcessed {len(json_files)} files")
    print(f"Generated {total_questions} questions")
    print(f"Output saved to: {output_json}")
    
    # Print query type distribution
    query_types = {}
    for image_data in grouped_results:
        for q in image_data['questions']:
            qt = q['query_type']
            query_types[qt] = query_types.get(qt, 0) + 1
    
    print("\nQuery type distribution:")
    for qt, count in sorted(query_types.items()):
        print(f"  {qt}: {count}")

if __name__ == "__main__":

    annotations_folder = 'synthetic_dataset_generation/output/annotations'
    annotations_masked_folder = 'synthetic_dataset_generation/output/masked/annotations'
    output_json = 'queries.json'
    compile_json(annotations_folder, annotations_masked_folder, output_json)