# Copyright (c) 2025 MKM Research Labs. All rights reserved.
# 
# This software is provided under license by MKM Research Labs. 
# Use, reproduction, distribution, or modification of this code is subject to the 
# terms and conditions of the license agreement provided with this software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



def replace_terms(text: str) -> str:
    """Replace specified terms with 'MKM' and convert American spelling to British English"""
    
    # Original replacements
    replacements = [
        "Kerr Shearer"
    ]
    
    # American to British spelling replacements
    spelling_replacements = {
        "behavior": "behaviour",
        "color": "colour",
        "labor": "labour",
        "center": "centre",
        "liter": "litre",
        "meter": "metre",
        "fiber": "fibre",
        "traveler": "traveller",
        "modeling": "modelling",
        "canceled": "cancelled",
        "fulfill": "fulfil",
        "enrollment": "enrolment",
        "catalog": "catalogue",
        "dialog": "dialogue",
        "analyze": "analyse",
        "paralyze": "paralyse",
        "defense": "defence",
        "license": "licence",
        "anemia": "anaemia",
        "pediatrician": "paediatrician",
        "esthet": "aesthet",
        "gray": "grey",
        "jewelry": "jewellery",
        "plow": "plough",
        "skeptical": "sceptical",
        "modeling":"modelling"
    }
    
    # Replace specified terms with 'MKM'
    for term in replacements:
        text = text.replace(term, "MKM")
        text = text.replace(term.lower(), "MKM")
        text = text.replace(term.upper(), "MKM")
    
    # Replace American spelling with British spelling
    for american, british in spelling_replacements.items():
        text = text.replace(american, british)
        text = text.replace(american.capitalize(), british.capitalize())
    
    return text
