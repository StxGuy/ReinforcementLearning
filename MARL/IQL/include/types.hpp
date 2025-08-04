#ifndef TYPES_H
#define TYPES_H

#include <vector>

/*----------------------------------- 
 * Enumeration for possible actions.
 *-----------------------------------*/
enum class Action{
    goNorth,
    goSouth,
    goEast,
    goWest,
    doNothing
};

/*----------------------
 * 2D vector structure.
 *----------------------*/
struct Vector2D {
    int x, y;
    
    // Constructor
    constexpr Vector2D(int x = 0, int y = 0) noexcept: x(x), y(y) {}
    
    // Displace vector in 2D
    [[nodiscard]] constexpr Vector2D move(Action act) const noexcept {
        switch(act) {
            case Action::goNorth: return{x, y + 1};
            case Action::goSouth: return{x, y - 1};
            case Action::goEast:  return{x + 1, y};
            case Action::goWest:  return{x - 1, y};
            default:              return{x, y};
        }
    }
    
    // Compare two vectors
    [[nodiscard]] constexpr bool operator==(const Vector2D& other) const noexcept {
        return x == other.x && y == other.y;
    }
    
    // Map a vector to a scalar
    int idx(int Lx) {
        return y*Lx + x;
    }
};

/*----------------------------
 * Structure for observation.
 *----------------------------*/
struct Observation {
    Vector2D    self_position;
    int         move_count;
    Vector2D    goal;
    std::vector<Vector2D> other_positions;
};

#endif  /* !TYPES_H */
